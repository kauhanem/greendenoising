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

    EPOCHS = 100
    folds = 10
  
    lf = nn.MSELoss()
    kf = KFold(n_splits=folds)

    tracker = CarbonTracker(epochs=EPOCHS, interpretable = False,
                            update_interval= 1, epochs_before_pred=0,
                            monitor_epochs=EPOCHS)

    print('--------------------------------')
    for f, (testIds, trainIds) in enumerate(kf.split(dataset)):
        print(f'FOLD {f}/{folds}')
        print('--------------------------------')

        trainSampler = tutils.SubsetRandomSampler(trainIds)
        testSampler = tutils.SubsetRandomSampler(testIds)

        trainLoader = tutils.DataLoader(dataset, sampler=trainSampler)
        testLoader = tutils.DataLoader(dataset, sampler=testSampler)

        model.apply(randomizeWeights)
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(EPOCHS):
            tracker.epoch_start()

            print(f'\nStarting epoch {epoch+1}/{EPOCHS}\n')
            currentLoss = 0.0

            for i, data in enumerate(trainLoader, 0):         
                inputs, targets = data

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = lf(outputs, targets)
                loss.backward()
                
                optimizer.step()

                currentLoss += loss.item()
                if i % 20 == 19:
                    print('Loss after mini-batch %5d: %.3f' %(i + 1, currentLoss / 20))
                    currentLoss = 0.0
            
            tracker.epoch_end()

        # Save each fold
        print('Training process has finished. Saving trained model.')

        savePath = f'./{name}-fold-{f}.pth'
        torch.save(model.state_dict(), savePath)
        
        # Start evaluation
        print('Starting evaluation')

        totalPSNR, currentPSNR = 0, 0
        with torch.no_grad():
            for i, data in enumerate(testLoader, 0):
                input, target = data
                output = model(input)

                currentPSNR = PSNR(target, output)
                totalPSNR = totalPSNR + currentPSNR
            
            avgPSNR = totalPSNR / i

            print(f"Average PSNR for fold {f}: {avgPSNR} dB")
            print('--------------------------------')
            results[f] = avgPSNR
    
    print(f"K-FOLD CROSS VALIDATION RESULTS FOR {folds} FOLDS")
    print('--------------------------------')

    sum = 0.0

    for key, value in results.items():
        print(f"Fold {key}: {value} dB")
        sum += value
    
    print(f"Avg: {sum/len(results.items())} dB")

    tracker.stop()

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