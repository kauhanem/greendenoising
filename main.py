import torch
import torch.nn as nn
import torch.utils.data as tutils

from noisyds import NoisyDataset

from sklearn.model_selection import KFold
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.io import imshow, imshow_collection, show

from fastonn import SelfONN2d
from dncnn import DnCNN
from scnn import sCNN
from onn import ONN

from carbontracker.tracker import CarbonTracker

import numpy as np
import matplotlib.pyplot as plt
import glob

def randomizeWeights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, SelfONN2d):
        # print(f'Xavier initializing trainable parameters of layer = {model}')
        nn.init.xavier_uniform_(model.weight)

def trainAndMeasure(model, dataset, name, epochs=100, folds=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    results = {}
    models = []
  
    lf = nn.MSELoss()
    kf = KFold(n_splits=folds)

    trainTracker = CarbonTracker(epochs = 1, interpretable = False,
                            update_interval = 1, epochs_before_pred=0,
                            monitor_epochs = 1, verbose=0)
    
    #evalTracker = CarbonTracker(epochs=EPOCHS, interpretable = False,
    #                        update_interval= 1, epochs_before_pred=0,
    #                        monitor_epochs=EPOCHS)

    optimizer = torch.optim.Adam(model.parameters())
    
    print('--------------------------------')
    for f, (testIds, trainIds) in enumerate(kf.split(dataset), 1):
        print(f'MODEL {name}, FOLD {f}/{folds}')
        print('--------------------------------')

        trainSize = len(trainIds)
        testSize = len(testIds)

        trainSampler = tutils.SubsetRandomSampler(trainIds)
        testSampler = tutils.SubsetRandomSampler(testIds)

        trainLoader = tutils.DataLoader(dataset, sampler=trainSampler)
        testLoader = tutils.DataLoader(dataset, sampler=testSampler)
        
        bestPSNR = 0

        trainTracker.epoch_start()

        for r in range(3):
            modelRand = model.apply(randomizeWeights)

            psnrPerEpoch = []

            for epoch in range(epochs):

                print(f'\nInitialization {r+1}/3, starting epoch {epoch+1}/{epochs}\n')

                currentLoss = 0.0
                epochPSNR = 0

                for i, data in enumerate(trainLoader, 0):         
                    input, target = data

                    input = input.to(device)
                    target = target.to(device)

                    optimizer.zero_grad()

                    output = modelRand(input).to(device)

                    loss = lf(output, target)
                    loss.backward()

                    optimizer.step()

                    currentLoss += loss.item()
                    
                    epochPSNR += PSNR(target.cpu().detach().numpy(), output.cpu().detach().numpy())
               
                    if (i+1) % (trainSize/10) == 0:
                        print('After mini-batch %2d/10, loss %.3f, PSNR: %.3f dB'
                               % ((i+1)/(trainSize/10), currentLoss / (trainSize/10), epochPSNR / (i+1)))
                        currentLoss = 0.0

                        if epoch == epochs-1:
                            imshow_collection([input.cpu().detach().squeeze(),
                                                output.cpu().detach().squeeze(),
                                                target.cpu().detach().squeeze()],
                                                cmap="gray")
                            show()
                
                psnrPerEpoch.append(epochPSNR / trainSize)
            
            if psnrPerEpoch[-1] > bestPSNR:
                bestPSNR = psnrPerEpoch[-1]
                bestModel = (modelRand, psnrPerEpoch, r+1)

        models.append(bestModel)

        trainTracker.epoch_end()

        # Start evaluation
        print(f'\nStarting evaluation on best randomized model {bestModel[2]}/3\n')

        totalPSNR, currentPSNR = 0, 0
        with torch.no_grad():

            for i, data in enumerate(testLoader, 0):
                if (i+1) % (testSize/10) == 0:
                    print('After mini-batch %2d/10, PSNR: %.3f dB' %((i+1)/(testSize/10), totalPSNR / (i+1)))

                input, target = data

                input = input.to(device)
                target = target.to(device)

                output = bestModel[0](input)

                currentPSNR = PSNR(target.cpu().detach().numpy(), output.cpu().detach().numpy())
                totalPSNR += currentPSNR
            
            avgPSNR = totalPSNR / testSize

            print("\nFold %s: %.3f dB\n" % (f, avgPSNR))
            print('--------------------------------')
            results[f] = avgPSNR
    

    print(f"MODEL {name}: K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS")
    print('--------------------------------')

    sum = 0.0

    for key, value in results.items():
        print("Fold %s: %.3f dB" % (key, value))
        sum += value
    
    print("\nAvg: %.3f dB" % (sum/len(results.items())))

    # Save best model
    print("\nSaving all folds' models")

    for i,m in enumerate(models, 1):
        savePath = f'./{name}-fold-{i}.pth'
        torch.save(m[0].state_dict(), savePath)

    return models

def main():
    # Load training dataset
    path = "Train/1/"
    data = NoisyDataset(path)

    #modelShallow = sCNN()
    #resultsShallow = trainAndMeasure(modelShallow, data, "shallowCNN")

    modelDeep = DnCNN()
    resultsDeep = trainAndMeasure(modelDeep, data, "deepCNN")

    # modelONN = ONN()
    # resultsONN = trainAndMeasure(modelONN, data, "ONN")

    # # Load testing dataset
    # noisyPath8 = "test/8/imp"
    # truthPath8 = "test/8/org"

if __name__ == '__main__':
    main()