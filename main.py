import torch
import torch.nn as nn
import torch.utils.data as tutils
from fastonn import SelfONN2d

from noisyds import NoisyDataset

from sklearn.model_selection import KFold
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.io import imshow, imshow_collection

from dncnn import DnCNN
from scnn import sCNN
from onn import ONN

from carbontracker.tracker import CarbonTracker

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def randomizeWeights(model):
    if isinstance(model, nn.Conv2d) or isinstance(model, SelfONN2d):
        print(f'Xavier initializing trainable parameters of layer = {model}')
        nn.init.xavier_uniform_(model.weight)

def trainAndMeasure(model, dataset, name):
    results = {}
    models = []

    EPOCHS = 1
    folds = 10
  
    lf = nn.MSELoss()
    kf = KFold(n_splits=folds)

    trainTracker = CarbonTracker(epochs=EPOCHS, interpretable = False,
                            update_interval= 1, epochs_before_pred=0,
                            monitor_epochs=EPOCHS)
    
    testTracker = CarbonTracker(epochs=EPOCHS, interpretable = False,
                            update_interval= 1, epochs_before_pred=0,
                            monitor_epochs=EPOCHS)
    
    print('--------------------------------')
    for f, (testIds, trainIds) in enumerate(kf.split(dataset)):
        print(f'MODEL {name}, FOLD {f}/{folds-1}')
        print('--------------------------------')

        trainSize = len(trainIds)
        testSize = len(testIds)

        trainSampler = tutils.SubsetRandomSampler(trainIds)
        testSampler = tutils.SubsetRandomSampler(testIds)

        trainLoader = tutils.DataLoader(dataset, sampler=trainSampler)
        testLoader = tutils.DataLoader(dataset, sampler=testSampler)

        randomizedModels = []

        for _ in range(3):
            model.apply(randomizeWeights)
            optimizer = torch.optim.Adam(model.parameters())

            for epoch in range(EPOCHS):
                trainTracker.epoch_start()

                print(f'\nStarting epoch {epoch+1}/{EPOCHS}\n')

                currentLoss = 0.0
                epochPSNR = 0

                for i, data in enumerate(trainLoader, 0):         
                    input, target = data

                    optimizer.zero_grad()

                    outputs = model(input)

                    loss = lf(outputs, target)
                    loss.backward()

                    epochPSNR += PSNR(target.numpy(), output.numpy())
                    
                    optimizer.step()

                    currentLoss += loss.item()
                    if (i+1) % (trainSize/10) == 0:
                        print('Loss after mini-batch %2d/10: %.3f' %((i+1)/(trainSize/10), currentLoss / (trainSize/10)))
                        currentLoss = 0.0

                
                
                trainTracker.epoch_end()

            randomizedModels.append(model)

        models.append(model)

        # # Save each fold
        # print('\nTraining process has finished. Saving trained model.\n')

        # savePath = f'./{name}-fold-{f}.pth'
        # torch.save(model.state_dict(), savePath)
        
        # Start evaluation
        print('\nStarting evaluation\n')

        totalPSNR, currentPSNR = 0, 0
        with torch.no_grad():
            for i, data in enumerate(testLoader, 0):
                if (i+1) % (testSize/10) == 0:
                    print('PSNR after mini-batch %2d/10: %.3f dB' %((i+1)/(testSize/10), totalPSNR / (i+1)))

                input, target = data
                output = model(input)

                currentPSNR = PSNR(target.numpy(), output.numpy())
                totalPSNR += currentPSNR
            
            avgPSNR = totalPSNR / testSize

            print("\nFold %s: %.3f dB\n" % (f, avgPSNR))
            print('--------------------------------')
            results[f] = avgPSNR
    
    trainTracker.stop()
    testTracker.stop()

    print(f"MODEL {name}: K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS")
    print('--------------------------------')

    sum = 0.0
    topValue = 0
    topKey = 0

    for key, value in results.items():
        print("Fold %s: %.3f dB" % (key, value))

        if value > topValue:
            topValue = value
            topKey = key

        sum += value
    
    print("Avg: %.3f dB" % (sum/len(results.items())))
    print("Max: %.3f dB (Fold %s)" % (topValue, topKey))

    # Save best model
    print('Saving best model.\n')

    savePath = f'./{name}-fold-{topKey}.pth'
    torch.save(models[topKey].state_dict(), savePath)

def main():
    # Load training dataset
    path = "Train/1/"
    data = NoisyDataset(path)

    # TODO: sCNN model training
    modelShallow = sCNN()
    resultsShallow = trainAndMeasure(modelShallow, data, "shallowCNN")

    # modelDeep = DnCNN()
    # resultsShallow = trainAndMeasure(modelDeep, data, "deepCNN")

    # modelONN = ONN()
    # resultsShallow = trainAndMeasure(modelONN, data, "ONN")

    # # Load testing dataset
    # noisyPath8 = "test/8/imp"
    # truthPath8 = "test/8/org"

if __name__ == '__main__':
    main()