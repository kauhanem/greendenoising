import torch
import torch.nn as nn
import torch.utils.data as tutils
from torcheval.metrics import PeakSignalNoiseRatio as psnr

from noisyds import NoisyDataset

from sklearn.model_selection import KFold

from fastonn import SelfONN2d
from dncnn import DnCNN
from scnn import sCNN
from onn import ONN

from carbontracker.tracker import CarbonTracker
from carbontracker import parser

import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
from time import time

def xavierInit(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, SelfONN2d):
        nn.init.xavier_uniform_(model.weight)

def test(models, dataset, name, plots=5):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        workers = 12
        pinMem = True
        torch.backends.cudnn.benchmark = True
        
    else:
        dev = torch.device("cpu")
        workers = 1
        pinMem = False
    
    metric = psnr(device=dev)

    testLoader = tutils.DataLoader(dataset, num_workers=workers, pin_memory=pinMem)

    testSize = dataset.__len__()
    numModels = len(models)

    result = np.zeros(numModels)

    modelTotal = 0
    
    outs = []
    pics = []

    print('----------------------------------------------------------')
    print(f'STARTING MODEL {name} TESTING. START TIME: {datetime.now().strftime("%H.%M:%S")}')
    print(f"FOLDS {numModels} | TESTSET SIZE {testSize}")
    print(f"USING DEV: {torch.cuda.get_device_name(0)}")
    print('----------------------------------------------------------')

    for m, model in enumerate(models, 0):
        modelStart = time()
        print(f'MODEL {name} | FOLD {m+1}/{numModels}')
        
        model = model.to(dev)
        model.eval()

        testPSNR = 0

        modelOuts = []

        with torch.no_grad():
            for j, data in enumerate(testLoader, 1):
                img, org = data

                img, org = img.to(dev), org.to(dev)

                out = model(img).to(dev)

                metric.update(out, org)
                currentPSNR = metric.compute()

                testPSNR += currentPSNR
            
                if j % (testSize/plots) == 0:
                    modelOuts.append(out.cpu().detach().squeeze())
                    if m == 0:
                        pics.append((img.cpu().detach().squeeze(),
                                     org.cpu().detach().squeeze()))

        outs.append(modelOuts)

        result[m] = testPSNR / j
        print(f"PSNR: {result[m]:.2f} dB")

        elapsed = time() - modelStart
        modelTotal += elapsed
        print(f"Model {m} testing took {elapsed/60:.2f} min (total {modelTotal/60:.2f} min)\n")

    return np.average(result), outs, pics
    
def train(model, dataset, name, experimentName, epochs=100, folds=10, batch=20, inits=3):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        workers = 20
        pinMem = True
        torch.backends.cudnn.benchmark = True
        
    else:
        dev = torch.device("cpu")
        workers = 1
        pinMem = False
    
    print('----------------------------------------------------------')
    print(f'STARTING MODEL {name} TRAINING. START TIME: {datetime.now().strftime("%H.%M:%S")}')
    print(f"FOLDS {folds} | EPOCHS {epochs} | INITS {inits}")
    print(f"USING DEV: {torch.cuda.get_device_name(0)}")
    print('----------------------------------------------------------')

    model = model.to(dev)

    kf = KFold(n_splits=folds)
    lf = nn.MSELoss().to(dev)
    optimizer = torch.optim.Adam(model.parameters())
    metric = psnr(device=dev)

    models = []
    PPEs = torch.zeros(folds, epochs)
    
    foldTotal = 0
     
    for f, (validIds, trainIds) in enumerate(kf.split(dataset), 1):
        foldStart = time()

        trainSampler = tutils.SubsetRandomSampler(trainIds)
        validSampler = tutils.SubsetRandomSampler(validIds)

        trainLoader = tutils.DataLoader(dataset, sampler=trainSampler,
                                        batch_size=batch, num_workers=workers,
                                        pin_memory=pinMem)
        validLoader = tutils.DataLoader(dataset, sampler=validSampler,
                                        batch_size=batch, num_workers=workers,
                                        pin_memory=pinMem)
        
        bestPSNR = -1

        # Train and evaluate n=inits Xavier initialized models
        for r in range(inits):
            modelRand = model.apply(xavierInit)

            psnrPerEpoch = torch.zeros(epochs)

            epochTotal = 0

            for epoch in range(epochs):
                epochStart = time()
                print(f'\nMODEL {name} | FOLD {f}/{folds} | INIT {r+1}/{inits} | EPOCH {epoch+1}/{epochs}')

                modelRand.train(True)

                totalLoss = 0

                # Training
                for i, data in enumerate(trainLoader, 1):         
                    img, org = data

                    img, org = img.to(dev), org.to(dev)

                    optimizer.zero_grad()

                    out = modelRand(img).to(dev)

                    loss = lf(out, org)
                    loss.backward()

                    optimizer.step()

                    totalLoss += loss.item()
                    
                print(f"Loss: {totalLoss/i:.3f}")
                
                # Validation
                modelRand.eval()
                epochPSNR = 0
                
                with torch.no_grad():
                    for i, data in enumerate(validLoader, 1):
                        img, org = data

                        img, org = img.to(dev), org.to(dev)

                        out = modelRand(img).to(dev)

                        metric.update(out, org)
                        epochPSNR += metric.compute()
                       
                avgPSNR = epochPSNR / i

                print(f"PSNR {avgPSNR:.3f} dB")
                psnrPerEpoch[epoch] = avgPSNR

                elapsed = time() - epochStart
                epochTotal += elapsed
                print(f"Execution took {elapsed/60:.2g} min (total {epochTotal/60:.2g} min)")

            if psnrPerEpoch[-1] > bestPSNR:
                bestPSNR = psnrPerEpoch[-1]

                bestModel = modelRand
                bestPPE = psnrPerEpoch
                bestInit = r+1

        print(f"\nFold {f} best init: {bestInit}/{inits}, PSNR: {bestPPE[-1]:.2f} dB")

        models.append(bestModel)
        PPEs[f-1] = bestPPE

        elapsed = time() - foldStart
        foldTotal += elapsed
        print(f"Fold {f} execution took {elapsed/60:.2f} min (total {foldTotal/60:.2f} min)")

    print('\n----------------------------------------------------------')
    print(f"MODEL {name}: K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS:")

    for ifold in range(folds):
        print(f"Fold {ifold+1}: {PPEs[ifold][-1]:.2f} dB")

    print('----------------------------------------------------------')

    # Calculate avg PSNR per epoch over all folds
    avgPSNRPerEpoch = torch.mean(PPEs, 0, True).numpy()[0]
    
    # Save all models
    print("\nSaving all folds' weights\n")
       
    for i,m in enumerate(models, 1):
        savePath = f'{experimentName}/{name}/fold{i}.pth'
        torch.save(m.state_dict(), savePath)
 
    return models, avgPSNRPerEpoch

def trainAndTest(model, name, experimentName, dataTrain, dataTest, quick, batch=20):
    if not os.path.isdir(f"{experimentName}/{name}"):
        os.mkdir(f"{experimentName}/{name}")

    if not os.path.isdir(f"{experimentName}/{name}/train"):
        os.mkdir(f"{experimentName}/{name}/train")

    if not os.path.isdir(f"{experimentName}/{name}/test"):
        os.mkdir(f"{experimentName}/{name}/test")

    trainTracker = CarbonTracker(epochs = 1, monitor_epochs = -1,
                                  update_interval = 1, epochs_before_pred=0, verbose=0,
                                  log_dir=f'{experimentName}/{name}/train', log_file_prefix=f"{name}")
    
    testTracker = CarbonTracker(epochs = 1, monitor_epochs = -1,
                                  update_interval = 1, epochs_before_pred=0, verbose=0,
                                  log_dir=f'{experimentName}/{name}/test', log_file_prefix=f"{name}")
    
    start = time()

    # Start training energy tracking
    trainTracker.epoch_start()

    if quick:
        models, trainPSNR = train(model, dataTrain, name, experimentName, batch=batch,
                                            epochs=10, folds=2, inits=2)
    else:
        models, trainPSNR = train(model, dataTrain, name, experimentName, batch=batch)
    
    # Stop training energy tracking
    trainTracker.epoch_end()

    trainEnd = time()
    trainTime = trainEnd - start

    print(f"MODEL {name} TRAINING TOOK {(trainTime)/60:.2f} MIN\n")

    # Start testing energy tracking
    testTracker.epoch_start()

    testPSNR, testOuts, testPics = test(models, dataTest, name)

    # Stop testing energy tracking
    testTracker.epoch_end()

    testTime = time() - trainEnd
    
    print(f"MODEL {name} TESTING TOOK {(testTime)/60:.2f} MIN\n")
    
    return trainPSNR, trainTime, testPSNR, testOuts, testPics, testTime

def main():
    quick = True

    scnn = True
    dncnn = True
    onn = True

    experimentName = f"experiments/{datetime.now().strftime('%Y-%-m-%-d-%H%M')}"
    if not os.path.isdir("experiments"):
        os.mkdir("experiments")
    if not os.path.isdir(experimentName):
        os.mkdir(experimentName)

    # Load training dataset
    path = "Train/2/"
    dataTrain = NoisyDataset(path)

    # Load testing dataset
    path = "Test/10/"
    dataTest = NoisyDataset(path)
    
    if quick:    
        dataTrain = tutils.Subset(dataTrain, list(range(100)))
        dataTest = tutils.Subset(dataTest, list(range(100)))

    names = []

    trainPSNRs = []
    testPSNRs = []
    
    outs = []

    trainEnergies = []
    testEnergies = []
    
    trainTimes = []
    testTimes = []

    # Shallow CNN
    if scnn:
        name = "CNN"
        name = name + "q" if quick else name
        names.append(name)

        trainPSNR, trainTime, testPSNR, testOuts, testPics, testTime = trainAndTest(sCNN(), name, experimentName,
                                                               dataTrain, dataTest, quick)

        trainPSNRs.append(trainPSNR)
        trainTimes.append(trainTime)
        testPSNRs.append(testPSNR)
        outs.append(testOuts)
        testTimes.append(testTime)

    # Deep CNN
    if dncnn:
        name = "DnCNN"
        name = name + "q" if quick else name
        names.append(name)

        trainPSNR, trainTime, testPSNR, testOuts, testPics, testTime = trainAndTest(DnCNN(), name, experimentName,
                                                               dataTrain, dataTest, quick)

        trainPSNRs.append(trainPSNR)
        trainTimes.append(trainTime)
        testPSNRs.append(testPSNR)
        outs.append(testOuts)
        testTimes.append(testTime)
  
    # SelfONN
    if onn:
        name = "ONN"
        name = name + "q" if quick else name
        names.append(name)

        trainPSNR, trainTime, testPSNR, testOuts, testPics, testTime = trainAndTest(ONN(), name, experimentName,
                                                               dataTrain, dataTest, quick)
        
        trainPSNRs.append(trainPSNR)
        trainTimes.append(trainTime)
        testPSNRs.append(testPSNR)
        outs.append(testOuts)
        testTimes.append(testTime)

    # Results plotting
    print(f"Plotting and printing test images of models {names}")

    # Print test images
    numModels = len(names)
    numPics = numModels+2
    numFolds = len(outs[0])

    for f in range(numFolds):
        figImg = plt.figure(figsize=(8,8))
        plt.axis("off")
        
        for r in range(5):
            figImg.add_subplot(5, numPics, 1 + r*numPics)
            plt.axis("off")
            if r == 0:
                plt.title("Noisy")
            plt.imshow(testPics[r][0], cmap="gray")

            figImg.add_subplot(5, numPics, (r+1)*numPics)
            plt.axis("off")
            if r == 0:
                plt.title("Truth")
            plt.imshow(testPics[r][1], cmap="gray")

            for m in range(numModels):
                figImg.add_subplot(5, numPics, m + 2 + r*numPics)
                plt.axis("off")
                if r == 0:
                    plt.title(names[m])
                plt.imshow(outs[m][f][r], cmap="gray")

            figImg.subplots_adjust(
                wspace=0,
                hspace=0
            )

        figImg.savefig(f"{experimentName}/images_fold{f+1}.png", bbox_inches='tight', dpi=300)
        
    # Plot PSNR per epoch
    print(f"Plotting and printing PSNR per epoch of models {names}")

    e = np.arange(1, len(trainPSNR)+1)

    col1 = ['darkorange','darkred','darkgreen']
    col2 = ['bisque','red','lightgreen']

    figPlot = plt.figure(figsize=(8,8))
    plt.title(f"Impulse Denoising")

    plt.ylabel(f"PSNR (dB)")
    plt.xlabel(f"Epochs")

    plt.grid(color='k', linestyle='-', linewidth=1, alpha=0.25)

    legend = []

    for m, y in enumerate(trainPSNRs, 0):
        plt.plot(e, y, color=col1[m])
        legend.append(f"[{y[-1]:.2f} dB] {names[m]} Training")
    
    for m, p in enumerate(testPSNRs, 0):
        plt.plot(len(y), p, color=col1[m], marker='x')
        legend.append(f"[{p:.2f} dB] {names[m]} Test")

    plt.xticks(e, e)

    plt.legend(legend)

    figPlot.savefig(f"{experimentName}/PSNR.png", bbox_inches='tight', dpi=300)

    # Plot energy use
    print(f"Plotting and printing energy use of architectures {names}")

    legend = []
    conversion = 3.6e6

    w = 0.4
    x = np.arange(numModels)

    for name in names:
        log = parser.parse_all_logs(log_dir=f"{experimentName}/{name}/train")

        # carbontracker gives results in joules, converted here to kWh
        trainkWh = log[0]['components']['gpu']['avg_energy_usages (J)'][0][0] / conversion
        testkWh = log[0]['components']['gpu']['avg_energy_usages (J)'][1][0] / conversion

        trainEnergies.append(trainkWh)
        testEnergies.append(testkWh)
    
    for i, name in enumerate(names, 0):
        legend.append(f"[{trainEnergies[i]:.2E} kWh] {name} Training")
        
    for i, name in enumerate(names, 0):
        legend.append(f"[{testEnergies[i]:.2E} kWh] {name} Testing")

    figEnergy = plt.figure(figsize=(8,8))
    plt.title(f"Energy Use per Architecture")
    plt.ylabel("Energy use (kWh)")

    plt.bar(x - w/2, trainEnergies, w, label=names, color=col1)
    plt.bar(x + w/2, testEnergies, w, label=names, color=col2)

    plt.xticks(x, names)

    plt.legend(legend)

    figEnergy.savefig(f"{experimentName}/Energy_use.png", bbox_inches='tight', dpi=300)

    # Plot compute times
    print(f"Plotting and printing training compute time of models {names}")

    trainTimes = np.array(trainTimes) / 60
    testTimes = np.array(testTimes) / 60

    legend = []

    for name in names:
        legend.append(f"[{trainTimes[-1]:.2f} min] {name} Training")

    for name in names:
        legend.append(f"[{testTimes[-1]:.2f} min] {name} Testing")

    figCompute = plt.figure(figsize=(8,8))
    plt.title(f"Compute Time per Architecture")
    plt.ylabel("Compute time (min)")

    plt.bar(x - w/2, trainTimes, w, label=names, color=col1)
    plt.bar(x + w/2, testTimes, w, label=names, color=col2)

    plt.xticks(x, names)

    plt.legend(legend)

    figCompute.savefig(f"{experimentName}/Compute_time.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()