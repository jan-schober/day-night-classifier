import torch
from torchvision import models
import cv2
import argparse
import glob
import json

def load_model():
    # loading imagenet pretrained model from torchvision models
    mbv2 = models.mobilenet_v2(pretrained=True)
    in_features = mbv2.classifier[1].in_features
    # replacing final FC layer of pretrained model with our FC layer having output classes = 2 for day/night
    mbv2.classifier[1] = torch.nn.Linear(in_features, 2)
    # Load trained model onto CPU
    mbv2.load_state_dict(torch.load('models/mbv2_best_model.pth', map_location=torch.device('cpu')))
    # Setting model to evaluation mode
    mbv2.eval()
    return mbv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--img', required=False, help='Path to image')
    ap.add_argument('-d', '--dataroot', required= False, help = 'Path to images')
    ap.add_argument('-f', '--format', required=False, help= 'Image format jpg oder png')
    args = vars(ap.parse_args())
    '''
    print(str(args['img']))
    if str(args['img']) is not None:
        # imagenet mean and std to normalize data
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        print('WHYY')
        # loading model
        model = load_model()

        # reading image
        ori_img = cv2.imread(str(args['img']))
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # resizing image to standard size
        img = cv2.resize(img, (500,500))
        # changing order of channels to (channel, height, width) format used by PyTorch
        img = torch.tensor(img).permute(2,0,1)
        # normalizing image
        img = img / 255.0
        img = (img - mean)/std
        img = img.unsqueeze(0)

        out = model(img)
        pred = torch.argmax(out)
        label = 'Day' if pred == 0 else 'Night'
        cv2.putText(ori_img, f'Prediction:{label}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
        #cv2.imshow('Prediction', ori_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('test.jpg', ori_img)
    '''
    if str(args['dataroot']) is not None:
        day_list = []
        night_list = []
        data_root_path = str(args['dataroot'])
        data_format = str(args['format'])
        img_list = glob.glob(data_root_path + '*'+ data_format)

        # imagenet mean and std to normalize data
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # loading model
        model = load_model()
        print('Number of files %d' % len(img_list))
        max_i = len(img_list)
        i = 0
        for file in img_list:
            i+=1
            print(f"{i}/{max_i}")
            ori_img = cv2.imread(file)
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            # resizing image to standard size
            img = cv2.resize(img, (500, 500))
            # changing order of channels to (channel, height, width) format used by PyTorch
            img = torch.tensor(img).permute(2, 0, 1)
            # normalizing image
            img = img / 255.0
            img = (img - mean) / std
            img = img.unsqueeze(0)

            out = model(img)
            pred = torch.argmax(out)

            if pred == 0:
                day_list.append(file)
            else:
                night_list.append(file)
            export_dic = {}
            export_dic['Day_Images'] = day_list
            export_dic['Night_Images'] = night_list

        n_day = len(day_list)
        n_night = len(night_list)
        print(f'Number of day images: {n_day}')
        print(f'Number of night images: {n_night}')
        with open('export.json', 'w') as f:
            json.dump(export_dic, f)


if __name__ == '__main__':
    main()
