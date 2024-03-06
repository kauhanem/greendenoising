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

def test(models, dataset, name):
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

    for i, model in enumerate(models, 0):
        modelStart = time()
        print(f'Starting model {name} {i+1}/{numModels} testing')
        
        model = model.to(dev)
        model.eval()

        testPSNR = 0

        with torch.no_grad():
            for j, data in enumerate(testLoader, 1):
                img, org = data

                img, org = img.to(dev), org.to(dev)

                out = model(img).to(dev)

                metric.update(out, org)
                currentPSNR = metric.compute()

                testPSNR += currentPSNR
            
                if j % (testSize/10) == 0:
                    fig = plt.figure()

                    plt.title(f"Model {name}")
                    plt.axis("off")
            
                    fig.add_subplot(1,3,1)
                    plt.title(f"Impulse")
                    plt.axis("off")
                    plt.imshow(img.cpu().detach().squeeze(),
                            cmap="gray")

                    fig.add_subplot(1,3,2)
                    plt.title(f"Output, PSNR: {currentPSNR:.2f} dB")
                    plt.axis("off")
                    plt.imshow(out.cpu().detach().squeeze(),
                            cmap="gray")

                    fig.add_subplot(1,3,3)
                    plt.title(f"Truth")
                    plt.axis("off")
                    plt.imshow(org.cpu().detach().squeeze(),
                            cmap="gray")
                    
                    plt.subplots_adjust(bottom=0,
                                        top=1,
                                        left=0,
                                        right=1,
                                        wspace=0,
                                        hspace=0)

                    plt.savefig(f"{name}/test_{j}.png", bbox_inches='tight',dpi=300)

        result[i] = testPSNR / testSize
        print(f"PSNR: {result[i]:.2f} dB")
        elapsed = time() - modelStart
        modelTotal += elapsed
        print(f"Model {i} execution took {elapsed/60:.2g} min (total {modelTotal/60:.2g} min)\n")

    return np.average(result)
    
def trainAndMeasure(model, dataset, name, epochs=100, folds=10, batch=1):
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        workers = 20
        pinMem = True
        torch.backends.cudnn.benchmark = True
        
    else:
        dev = torch.device("cpu")
        workers = 1
        pinMem = False
    
    print(f"Using dev: {torch.cuda.get_device_name(0)}")

    metric = psnr(device=dev)

    model = model.to(dev)

    lf = nn.MSELoss()
    kf = KFold(n_splits=folds)
    optimizer = torch.optim.Adam(model.parameters())

    models = []
    PPEs = []

    carbontracker = CarbonTracker(epochs = 1, interpretable = False,
                            update_interval = 1, epochs_before_pred=0,
                            monitor_epochs = 1, verbose=0)
    
    # Start energy tracking
    carbontracker.epoch_start()
    
    foldTotal = 0

    for f, (validIds, trainIds) in enumerate(kf.split(dataset), 1):
        foldStart = time()

        print('------------------------------------------------')
        print(f'MODEL {name}, FOLD {f}/{folds}, START: {datetime.now().strftime("%H.%M:%S")}')
        print('------------------------------------------------')

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

        # Train and evaluate three Xavier initialized models
        for r in range(3):
            modelRand = model.apply(xavierInit)

            psnrPerEpoch = []

            epochTotal = 0

            for epoch in range(epochs):
                epochStart = time()
                print(f'\nFold {f}/{folds}, init {r+1}/3, epoch {epoch+1}/{epochs}')

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
                    
                print(f"Loss: {totalLoss/trainSize:.3f}")
                
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
                       
                avgPSNR = epochPSNR / validSize

                print(f"PSNR {avgPSNR:.3f} dB")
                psnrPerEpoch.append(avgPSNR)

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
        PPEs.append(bestPPE)

        elapsed = time() - foldStart
        foldTotal += elapsed
        print(f"Fold {f} execution took {elapsed/60:.2g} min (total {foldTotal/60:.2g} min)\n")

    # End energy tracking
    carbontracker.epoch_end()

    print('------------------------------------------------')
    print(f"MODEL {name}: K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS\n")

    # Calculate avg PSNR per epoch over all folds
    avgPSNRPerEpoch = []

    for i in range(epochs):
        sum = 0
        
        for j in range(folds):
            if i == epochs - 1:
                print(f"Fold {j} PSNR: {PPEs[j][i]:.3f} dB")

            sum += PPEs[j][i].cpu().numpy()
        
        avgPSNRPerEpoch.append(sum/folds)
    print('------------------------------------------------')

    # Save all models
    print("\nSaving all models\n")

    if not os.path.isdir(name):
        os.mkdir(name)
        
    for i,m in enumerate(models, 1):
        savePath = f'{name}/fold-{i}.pth'
        torch.save(m.state_dict(), savePath)

    return models, avgPSNRPerEpoch

def main():

    # Load training dataset
    path = "Train/1/"
    dataTrain = NoisyDataset(path)
    dataTrain = tutils.Subset(dataTrain, list(range(40)))

    # Load testing dataset
    path = "Test/9/"
    dataTest = NoisyDataset(path)
    dataTest = tutils.Subset(dataTest, list(range(40)))

    # Shallow CNN
    modelShallow = sCNN()
    shallowModels, y1 = trainAndMeasure(modelShallow, dataTrain, "shallowCNN")
    # shallowModels, y1 = trainAndMeasure(modelShallow, dataTrain, "shallowCNNquick", epochs=10, folds=2)

    t1 = test(shallowModels, dataTest, "shallowCNN")

    # Deep CNN
    # deepTime = datetime.now().strftime("%H.%M:%S")
    # print(f"Deep CNN start: {deepTime}")
    #
    # modelDeep = DnCNN()
    # deepModels, y2 = trainAndMeasure(modelDeep, data, "deepCNN", batch=50)
    # deepModels, y2 = trainAndMeasure(modelDeep, data, "deepCNNquick", batch=5, epochs=10, folds=2)
    #
    # t2 = test(deepModels, dataTest, "deepCNN")

    # Fast ONN
    modelONN = ONN()
    onnModels, y3 = trainAndMeasure(modelONN, dataTrain, "ONN")
    #onnModels, y3 = trainAndMeasure(modelONN, dataTrain, "ONNquick", epochs=10, folds=2)

    t3 = test(onnModels, dataTest, "ONN")

    e = list(range(1, len(y1)+1))
    
    fig = plt.figure()

    plt.title(f"Impulse Denoising")

    plt.ylabel(f"PSNR (dB)")
    plt.xlabel(f"Epochs")

    plt.grid(color='k', linestyle='-', linewidth=1, alpha=0.25)

    plt.plot(e, y1,'r')
    plt.plot(len(y1), t1, 'rx')

    #plt.plot(e, y2,'b')
    #plt.plot(len(y2), t2, 'bx')

    plt.plot(e, y3,'k')
    plt.plot(len(y3), t3, 'kx')

    plt.legend([f'[{y1[-1]:.2f} dB] sCNN Training', f'[{t1:.2f} dB] sCNN Test',
                #f'[{y2[-1]:.2f} dB] DnCNN Training', f'[{t2:.2f} dB] DnCNN Test',
                f'[{y3[-1]:.2f} dB] ONN Training', f'[{t3:.2f} dB] ONN Test'])

    plt.savefig(f"result.png", bbox_inches='tight',dpi=300)

if __name__ == '__main__':
    main()