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

    print('------------------------------------------------')
    print(f'STARTING MODEL {name} TESTING. START TIME: {datetime.now().strftime("%H.%M:%S")}')
    print(f"FOLDS {numModels} | TESTSET SIZE {testSize}")
    print(f"USING DEV: {torch.cuda.get_device_name(0)}")
    print('------------------------------------------------')

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

                img, org = torch.unsqueeze(img.to(dev), 1), torch.unsqueeze(org.to(dev), 1)

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

        result[m] = testPSNR / testSize
        print(f"PSNR: {result[m]:.2f} dB")

        elapsed = time() - modelStart
        modelTotal += elapsed
        print(f"Model {m} execution took {elapsed/60:.2f} min (total {modelTotal/60:.2f} min)\n")

    return np.average(result), outs, pics
    
def trainAndMeasure(model, dataset, name, experimentName, epochs=100, folds=10, batch=1, inits=3):
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

    metric = psnr(device=dev)

    model = model.to(dev)

    lf = nn.MSELoss()
    kf = KFold(n_splits=folds)
    optimizer = torch.optim.Adam(model.parameters())

    models = []
    PPEs = torch.zeros(folds, epochs)

    carbontracker = CarbonTracker(epochs = 1, interpretable = False,
                            update_interval = 1, epochs_before_pred=0,
                            monitor_epochs = 1, verbose=0)
    
    # Start energy tracking
    carbontracker.epoch_start()
    
    foldTotal = 0

    for f, (validIds, trainIds) in enumerate(kf.split(dataset), 1):
        foldStart = time()

        trainSize = len(trainIds)
        validSize = len(validIds)

        trainSampler = tutils.SubsetRandomSampler(trainIds)
        validSampler = tutils.SubsetRandomSampler(validIds)

        trainLoader = tutils.DataLoader(dataset, sampler=trainSampler,
                                        batch_size=batch, num_workers=workers,
                                        pin_memory=pinMem)
        validLoader = tutils.DataLoader(dataset, sampler=validSampler,
                                        batch_size=batch, num_workers=workers,
                                        pin_memory=pinMem)
        
        bestPSNR = 0

        # Train and evaluate n Xavier initialized models
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

                    img, org = torch.unsqueeze(img.to(dev), 1), torch.unsqueeze(org.to(dev), 1)

                    optimizer.zero_grad()

                    out = modelRand(img).to(dev)

                    loss = lf(out, org)
                    loss.backward()

                    optimizer.step()

                    totalLoss += loss.item()
                    
                print(f"Loss: {totalLoss/trainSize:.3f}")
                
                # Validation
                modelRand.eval()
                epochPSNR = 0
                
                with torch.no_grad():
                    for i, data in enumerate(validLoader, 1):
                        img, org = data

                        img, org = torch.unsqueeze(img.to(dev), 1), torch.unsqueeze(org.to(dev), 1)

                        out = modelRand(img).to(dev)

                        metric.update(out, org)
                        epochPSNR += metric.compute()
                       
                avgPSNR = epochPSNR / validSize

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

        print(f"\nFold {f} best init: {bestInit}/3, PSNR: {bestPPE[-1]:.2f} dB")

        models.append(bestModel)
        PPEs[f-1] = bestPPE

        elapsed = time() - foldStart
        foldTotal += elapsed
        print(f"Fold {f} execution took {elapsed/60:.2f} min (total {foldTotal/60:.2f} min)")

    # End energy tracking
    carbontracker.epoch_end()

    print('----------------------------------------------------------')
    print(f"MODEL {name}: K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS\n")

    for ifold in range(folds):
        print(f"Fold {ifold+1}: {PPEs[ifold][-1]:.2f} dB")

    print('----------------------------------------------------------')

    # Calculate avg PSNR per epoch over all folds
    avgPSNRPerEpoch = torch.mean(PPEs, 0, True).numpy()[0]
    
    # Save all models
    print("\nSaving all models\n")
       
    for i,m in enumerate(models, 1):
        savePath = f'{experimentName}/{name}/fold{i}.pth'
        torch.save(m.state_dict(), savePath)
 
    return models, avgPSNRPerEpoch

def trainAndTest(model, name, experimentName, dataTrain, dataTest, quick, batchSize=1):
    start = time()

    if not os.path.isdir(f"{experimentName}/{name}"):
        os.mkdir(f"{experimentName}/{name}")
    
    if quick:
        models, trainPSNR = trainAndMeasure(model, dataTrain, name, experimentName, epochs=10, folds=2, batch=batchSize, inits=1)
    else:
        models, trainPSNR = trainAndMeasure(model, dataTrain, name, experimentName, batch=batchSize)

    testPSNR, testOuts, testPics = test(models, dataTest, name)

    print(f"Model {name} execution took {(time()-start)/60:.2f} min\n")
    
    return trainPSNR, testPSNR, testOuts, testPics

def main():
    quick = False

    scnn = True
    dncnn = True
    onn = True

    experimentName = f"experiment_{datetime.now().strftime('%b%d_%H%M')}"
    if not os.path.isdir(experimentName):
        os.mkdir(experimentName)

    # Load training dataset
    path = "Train/1/"
    dataTrain = NoisyDataset(path)

    # Load testing dataset
    path = "Test/9/"
    dataTest = NoisyDataset(path)
    
    if quick:    
        dataTrain = tutils.Subset(dataTrain, list(range(40)))
        dataTest = tutils.Subset(dataTest, list(range(40)))

    trainPSNRs = []
    energyUses = []
    testPSNRs = []
    outs = []
    names = []

    # Shallow CNN
    if scnn:
        name = "CNN"
        name = name + "q" if quick else name
        names.append(name)

        trainPSNR, testPSNR, testOuts, testPics = trainAndTest(sCNN(), name, experimentName, dataTrain, dataTest, quick)
        
        trainPSNRs.append(trainPSNR)
        testPSNRs.append(testPSNR)
        outs.append(testOuts)


    # Deep CNN
    if dncnn:
        name = "DnCNN"
        name = name + "q" if quick else name
        names.append(name)

        trainPSNR, testPSNR, testOuts, testPics = trainAndTest(DnCNN(), name, experimentName, dataTrain, dataTest, quick, batchSize=20)

        trainPSNRs.append(trainPSNR)
        testPSNRs.append(testPSNR)
        outs.append(testOuts)
  
    # Fast ONN
    if onn:
        name = "ONN"
        name = name + "q" if quick else name
        names.append(name)

        trainPSNR, testPSNR, testOuts, testPics = trainAndTest(ONN(), name, experimentName, dataTrain, dataTest, quick)
        
        trainPSNRs.append(trainPSNR)
        testPSNRs.append(testPSNR)
        outs.append(testOuts)

    # Results plotting
        
    # Print test images
    numPics = len(outs)+2
    numFolds = len(outs[0])

    for f in range(numFolds):
        figImg = plt.figure(figsize=(8,6))
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

            for m in range(numPics-2):
                figImg.add_subplot(5, numPics, m + 2 + r*numPics)
                plt.axis("off")
                if r == 0:
                    plt.title(names[m])
                plt.imshow(outs[m][f][r], cmap="gray")

            figImg.subplots_adjust(
                wspace=0,
                hspace=0
            )

        plt.savefig(f"{experimentName}/images_fold{f+1}.png", bbox_inches='tight', dpi=300)
        
    # Plot PSNR per epoch
    e = list(range(1, len(trainPSNR)+1))
    col = ['b','r','k']

    figPlot = plt.figure(figsize=(8,6))
    plt.title(f"Impulse Denoising")

    plt.ylabel(f"PSNR (dB)")
    plt.xlabel(f"Epochs")

    plt.grid(color='k', linestyle='-', linewidth=1, alpha=0.25)

    legend = []

    for m,y in enumerate(trainPSNRs,0):
        plt.plot(e, y, col[m])
        legend.append(f"[{y[-1]:.2f} dB] {names[m]} Training")
    
    for m,p in enumerate(testPSNRs,0):
        plt.plot(len(y), p, col[m] + 'x')
        legend.append(f"[{p:.2f} dB] {names[m]} Test")

    plt.legend(legend)

    plt.savefig(f"{experimentName}/PSNR_plots.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    main()