import glob, os
import cv2
import numpy as np
from PIL import Image
from generate_labels import pascal_voc_to_yolo
import PySimpleGUI as sg
import shutil

global coordinates, fore_g_image, dir_fg_img, list_, list_bg_img,obj_pt, fh, fw, dir_fg_img_list
coordinates = []
list_bg_img = []

def background_img_copy(bg_images, len_images):
    # print('len_images............', len_images)
    bs_nam = os.path.basename(bg_images)
    bg_img_read = cv2.imread(bg_images)
    for img_counter in range(0, len_images):
        # print('helloooooooooooo')
        cv2.imwrite('Dataset/list_image/'+bs_nam[:-4]+str(img_counter)+".jpg", bg_img_read)
        list_bg_img.append('Dataset/list_image/'+bs_nam[:-4]+str(img_counter)+".jpg")
    # print(list_bg_img)

def popup_wind():
    layout = [[sg.Text('Select the Object Image to be placed')],
                    # [sg.Combo(['1','2' ], size=(20, 1), key='object', default_value='0-non')],
                    [sg.Text("Object Path:")],
                    [sg.InputText(key='inputxt'), sg.FileBrowse()],
                    
                    # [sg.Text("Tajas")],
                    # [ sg.InputText(key='inputxt4'), sg.FileBrowse()],
                    
                    [sg.Button('Ok'), sg.Button('Cancel')]
                    ]
    # Set dimensions of the window
    window = sg.Window('Select the object to be placed', layout)
    event, values = window.read()
    window.close()
    
    dir_ = values['inputxt']
    # Tejas_dir = values['inputxt4']
    return [dir_]

list_ = []
def coordinates_on_click(event, x, y, flags, params, offset=5, object_dim=[40,40]):

    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(bg_images_, str('.'), ((x-4),y), font, 1, (255, 255, 0), 2)
        #cv2.rectangle(bg_images_, str('.'), (x,y), font, 1, (255, 255, 0), 8)
        cv2.imshow('image', bg_images_)
        list_.append(dir_fg_img)
        
        coord = [x,y]
        for i in range(0,4):
            coordinates.append(coord)
            #coordinates.append(coord)
        
# index = 0
imgs = glob.glob('Dataset/background_images/*.jpg')

for bg_images in imgs:
    dir_fg_img_list = glob.glob('Dataset/input/*/*', recursive=True)
    background_img_copy(bg_images, len(dir_fg_img_list))
    base_name = os.path.basename(bg_images)
    bg_images_ = cv2.imread(bg_images)
    H, W, _ = bg_images_.shape
    # print('Base Image Dim =',H,'x',W)
    # cv2.imshow('image', bg_images_)
    # cv2.waitKey(0)

    #cv2.destroyAllWindows()
    #dir_fg_img = popup_wind()
    
    dir_fg_img = dir_fg_img_list

    cv2.imshow('image', bg_images_)
    cv2.setMouseCallback('image', coordinates_on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # print("coordinates",coordinates)
    chunked_list = coordinates
    chunk_size = 4
    
    lists = []
    for chun in range(0, len(coordinates), chunk_size):
        lists.append(coordinates[chun:chun+chunk_size])
        
    # print("lists",lists)
    for kh in lists:
        for img, ll in zip(dir_fg_img, list_bg_img):
            
            #print('dir_fg_img',dir_fg_img)
            # for ll in list_bg_img:
                
            bg_base_name = os.path.basename(ll)
            
           # txt_file = open('Dataset/Dataset_output/'+bg_base_name[:-4]+'.txt', 'w')
            fg_base_name = os.path.basename(img)
            
            # if 'Mig29' in fg_base_name:
            #     cls = 4
            
            # if 'Rafale' in fg_base_name:
            #     cls = 2
                
            # if 'Mirage2000' in fg_base_name:
            #     cls = 3
                
            # if 'SU_30' in fg_base_name:
            #     cls = 1
                
            # if 'Tejas' in fg_base_name:
            #     cls = 0
                
            # print("fg_base_name", fg_base_name)
            fg_img = Image.open(img, 'r').convert("RGBA")
            bg_img = Image.open(ll, 'r').convert("RGBA")
                
            obj_img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            obj_img.paste(bg_img, (0, 0))
            fg_image_size = cv2.imread(img)
            
            fh, fw, f_ = fg_image_size.shape
            img_ = Image.open(img, 'r').convert("RGBA") 
            
            #try:
            obj_img.paste(img_, (kh[0][0] - (fw//2), kh[0][1]  - (fh//2)), mask=img_)
            obj_img_cv = np.array(obj_img)
            obj_img_cv_bgr = cv2.cvtColor(obj_img_cv, cv2.COLOR_RGB2BGR)
            # print('coordinates[0]',coordinates[0][0])
            bb = pascal_voc_to_yolo((kh[0][0] - (fw//2)), (kh[0][1]  - (fh//2)),
                                    (kh[0][0])+(fw//2), (kh[0][1])+ (fh//2), W, H)
            #cv2.rectangle(obj_img_cv_bgr, (kh[0][0], kh[0][1]), ((kh[0][0])+fw,(kh[0][1])+fh), (0, 0, 255), 1)
            txt_file = open('Dataset/output/'+bg_base_name[:-4]+'.txt', 'a')
            txt_file.write("name"+" "+str(bb[0])+' '+str(bb[1])+' '+str(bb[2])+' '+str(bb[3])+'\n')
            txt_file.close()

            cv2.imwrite('Dataset/output/'+bg_base_name,obj_img_cv_bgr)
                # i += 1
            # except:
            #     pass
            
        list_bg_img = glob.glob('C:/Users/Shahzaib/Desktop/Doing/datageneration/senerio_3/Dataset/output/*.jpg')        
    
    for cop in list_bg_img:
        b_n = os.path.basename(cop)
        os.rename(cop,'Dataset/output/senerio_1/'+b_n)
        os.rename(cop[:-4]+'.txt', 'Dataset/output/senerio_1/'+b_n[:-4]+".txt")


    list_bg_img.clear()
    dir_fg_img.clear()
    coordinates.clear() 
    