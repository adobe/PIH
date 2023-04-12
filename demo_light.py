#!/usr/bin/env python

import pygame
import pygame_gui
from pygame_gui.windows.ui_file_dialog import UIFileDialog
from pygame_gui.elements.ui_button import UIButton
from pygame.rect import Rect
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot
import torchvision.transforms as T
import torchvision.transforms.functional as F
import glob
import os
import PIL
from PIL import Image
import cv2

from model import Model_Composite_PL, Model_Composite

transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
transform = T.Compose([T.ToTensor()])
resize = T.Resize((512, 512))


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode
    ).convert()


window_dim = (1680, 1260)

pygame.init()
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
window_surface = pygame.display.set_mode(window_dim)

background = pygame.Surface(window_dim)
# background.fill(pygame.Color('#000000'))

manager = pygame_gui.UIManager(window_dim)
clock = pygame.time.Clock()

file_selection_button = UIButton(
    relative_rect=Rect(0, 0, 200, 100), manager=manager, text="Select Background"
)

foreground_selection_button = UIButton(
    relative_rect=Rect(400, 0, 200, 100), manager=manager, text="Select Foreground"
)


model_selection_button = UIButton(
    relative_rect=Rect(800, 0, 200, 100), manager=manager, text="Select Model"
)

run_button = UIButton(
    relative_rect=Rect(1200, 0, 200, 100), manager=manager, text="Harmonize the image!"
)


font = pygame.font.Font("freesansbold.ttf", 16)
text = font.render("No background selected", True, green, blue)
textRect = text.get_rect()
textRect.center = (200 // 2, 300 // 2)


text_fore = font.render("No foreground selected", True, green, blue)
textRect_fore = text_fore.get_rect()
textRect_fore.center = (1000 // 2, 300 // 2)


text_model = font.render("No Model selected", True, green, blue)
textRect_model = text_model.get_rect()
textRect_model.center = (1800 // 2, 300 // 2)


image_selected = False
fore_selected = False
model_selected = False

flag_file_open = False
flag_fore_open = False
flag_model_open = False


output_exist = False

#### Zongze
flag_dialog = False

save_ind = 0
scale_sm = 0.55

while 1:
    time_delta = clock.tick(60) / 1000.0
    pygame.key.set_repeat(0, 0)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit()

        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:

                if event.ui_element == file_selection_button:
                    file_selection = UIFileDialog(
                        rect=Rect(0, 0, 300, 300),
                        manager=manager,
                        allow_picking_directories=True,
                    )
                    flag_file_open = True

                if event.ui_element == foreground_selection_button:
                    fore_selection = UIFileDialog(
                        rect=Rect(0, 0, 300, 300),
                        manager=manager,
                        allow_picking_directories=True,
                    )
                    flag_fore_open = True

                if event.ui_element == model_selection_button:
                    model_selection = UIFileDialog(
                        rect=Rect(0, 0, 300, 300),
                        manager=manager,
                        allow_picking_directories=True,
                    )
                    flag_model_open = True

                if flag_file_open:
                    if event.ui_element == file_selection.ok_button:
                        flag_dialog = True
                        # print(file_selection.current_file_path)
                        carImg = pygame.image.load(file_selection.current_file_path)
                        size_carimg = carImg.get_size()

                        carImg = pygame.transform.scale(carImg, (512, 512))

                        # img = cv2.imread(
                        #     str(file_selection.current_file_path), cv2.IMREAD_UNCHANGED
                        # )

                        # # extract alpha channel
                        # alpha = img[:, :, 3]

                        # # threshold alpha channel
                        # alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]

                        # cv2.imwrite("object_alpha.png", alpha)

                        # carImg = pygame.image.load("object_alpha.png")

                        text = font.render(
                            str(file_selection.current_file_path).split("/")[-1],
                            True,
                            green,
                            blue,
                        )

                        image_selected = True
                        flag_file_open = False

                if flag_fore_open:
                    if event.ui_element == fore_selection.ok_button:
                        flag_dialog = True
                        # print(fore_selection.current_file_path)
                        # ForeIMG = pygame.image.load(fore_selection.current_file_path)
                        # ForeIMG = pygame.transform.scale(ForeIMG, (512, 512))

                        img = cv2.imread(
                            str(fore_selection.current_file_path), cv2.IMREAD_UNCHANGED
                        )
                        img = cv2.resize(img, (512, 512))

                        # # extract alpha channel
                        alpha = img[:, :, 3]

                        # # threshold alpha channel
                        # alpha = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]

                        cv2.imwrite("image_mask.png", alpha)

                        # carImg = pygame.image.load("object_alpha.png")

                        text_fore = font.render(
                            str(fore_selection.current_file_path).split("/")[-1],
                            True,
                            green,
                            blue,
                        )

                        if image_selected:
                            img_bg = Image.open(str(file_selection.current_file_path))
                            img_bg = img_bg.resize((512, 512))
                            img_mask = Image.open("image_mask.png")
                            img_foreground = Image.open(
                                str(fore_selection.current_file_path)
                            )
                            img_foreground = img_foreground.resize((512, 512))
                            img_composite = Image.composite(
                                img_foreground, img_bg, img_mask
                            )
                            img_composite.save("image_composite.png")

                            carImg = pilImageToSurface(img_composite)

                        fore_selected = True
                        flag_fore_open = False

                if flag_model_open:
                    if event.ui_element == model_selection.ok_button:
                        # print(model_selection.current_file_path)
                        
                        flag_dialog = True
                        
                        
                        text_model = font.render(
                            str(model_selection.current_file_path).split("/")[-1],
                            True,
                            green,
                            blue,
                        )
                        model_selected = True
                        flag_model_open = False

                if event.ui_element == run_button:
                    if model_selected and image_selected:

                        model_path = str(model_selection.current_file_path)
                        image_path = str(file_selection.current_file_path)
                        word = image_path.split("_")[-1]
                        Model = Model_Composite_PL(
                            dim=32,
                            masking=True,
                            brush=True,
                            maskoffset=0.65,
                            swap=True,
                            Vit_bool=False,
                            onlyupsample=True,
                            aggupsample=True,
                            light=True,
                            Eff_bool=True,
                        )
                        
                        print("num of parameters:",sum(p.numel() for p in Model.parameters()))
                        device = "cpu"
                        # print("Loading Model")
                        checkpoint = torch.load(model_path, map_location=device)
                        Model.load_state_dict(checkpoint["state_dict"])
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cudnn.deterministic = True
                        Model.to(device)
                        Model.eval()

                        image_bg = resize(Image.open(image_path))
                        image_composite = resize(Image.open("image_composite.png"))
                        image_mask = resize(Image.open("image_mask.png"))

                        # Load image

                        scale = 1
                        translate = [0, 0]

                        torch_bg = transform(image_bg).to(device)
                        torch_composite = transform(image_composite).to(device)
                        torch_mask = transforms_mask(image_mask).to(device)

                        torch_composite = F.affine(
                            torch_composite,
                            angle=0,
                            translate=[0, 0],
                            scale=scale,
                            shear=0,
                        )
                        torch_mask = F.affine(
                            torch_mask, angle=0, translate=[0, 0], scale=scale, shear=0
                        )

                        torch_composite = F.affine(
                            torch_composite,
                            angle=0,
                            translate=translate,
                            scale=1,
                            shear=0,
                        )
                        torch_mask = F.affine(
                            torch_mask, angle=0, translate=translate, scale=1, shear=0
                        )

                        torch_composite = (
                            torch_composite * torch_mask + (1 - torch_mask) * torch_bg
                        )

                        with torch.no_grad():
                            inter_composite, output_composite, par1, par2 = Model(
                                torch_bg[None, ...],
                                torch_composite[None, ...],
                                torch_mask[None, ...],
                            )

                        output_lr = T.ToPILImage()(output_composite[0, ...])
                        inter_lr = T.ToPILImage()(inter_composite[0, ...])

                        composite_lr = T.ToPILImage()(torch_composite)
                        image_gainmap = T.ToPILImage()(
                            (
                                Model.output_final[0, ...].cpu()
                                * torch_mask[0, ...].cpu()
                            ).repeat(3, 1, 1)
                        ).resize((256, 256))
                        # print(Model.output_final[0, ...].cpu().shape)
                        # print(output_lr)
                        curves = par2.cpu().detach().numpy()

                        red_curve = curves[0, 0, 0, 0, :]
                        green_curve = curves[0, 1, 0, :, 0]
                        blue_curve = curves[0, 2, :, 0, 0]

                        plt.figure()
                        plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                        plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                        plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                        plt.ylim(0, 1)
                        plt.legend(["Reg", "Green", "Blue"])
                        plt.title("Learned Color Curves")

                        plt.savefig("tmp.jpg")
                        plt.close()

                        curveplot = pygame.image.load("tmp.jpg")
                        curveplot = pygame.transform.scale(curveplot, (256, 256))
                        
                        ### Cubic fitting
                        global idxx
                        idxx = np.linspace(0, 1, 32)
                        p_red = np.polyfit(idxx,red_curve,3)
                        p_red = p_red[3] + p_red[2]*idxx +  p_red[1]*idxx*idxx + p_red[0]*idxx*idxx*idxx
                        
                        p_blue = np.polyfit(idxx,blue_curve,3)
                        p_blue = p_blue[3] + p_blue[2]*idxx +  p_blue[1]*idxx*idxx + p_blue[0]*idxx*idxx*idxx
                        
                        
                        p_green = np.polyfit(idxx,green_curve,3)
                        p_green = p_green[3] + p_green[2]*idxx +  p_green[1]*idxx*idxx + p_green[0]*idxx*idxx*idxx
                        
                        # print(p_red)
                        red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                        green_curve_plot = [(100 + (i/32)*260, 980 - p_green[i]*260) for i in range(32)]
                        blue_curve_plot = [(100 + (i/32)*260, 980 - p_blue[i]*260) for i in range(32)]
                        
    # pygame.draw.lines(window_surface, (255,0,0), False, [(100,980),(360,720)])
                        
                        red_control = [red_curve_plot[0],red_curve_plot[10],red_curve_plot[20],red_curve_plot[31]]
                        green_control = [green_curve_plot[0],green_curve_plot[10],green_curve_plot[20],green_curve_plot[31]]
                        blue_control = [blue_curve_plot[0],blue_curve_plot[10],blue_curve_plot[20],blue_curve_plot[31]]
                        
                        # print(red_control)
                        
                        # print(red_curve_plot)
                        if not output_exist:
                            file_selection_red = UIButton(
                                relative_rect=Rect(400, 700, 100, 50), manager=manager, text="Red"
                            )
                            file_selection_green = UIButton(
                                relative_rect=Rect(400, 820, 100, 50), manager=manager, text="Green"
                            )
                            file_selection_blue = UIButton(
                                relative_rect=Rect(400, 940, 100, 50), manager=manager, text="Blue"
                            )

                        output_exist = True
                        red_check = False
                        green_check = False
                        blue_check = False
                        flag_dialog = False
                        save_ind = 0
                        
                if output_exist:      
                    if event.ui_element == file_selection_red:
                        red_check = True
                        green_check = False
                        blue_check = False
                        
                        # print(red_check)
                    if event.ui_element == file_selection_green:
                        red_check = False
                        green_check = True
                        blue_check = False
                        
                    if event.ui_element == file_selection_blue:
                        red_check = False
                        green_check = False
                        blue_check = True                                      
                    
                        
                    
        if event.type == pygame.MOUSEBUTTONDOWN:
            drag = False
            if output_exist:
                if red_check:
                    xMouse = event.pos[0]
                    yMouse = event.pos[1]
                    # print(xMouse,yMouse)
                    for ix, points_red in enumerate(red_control):
                        if ((xMouse-points_red[0])**2 + (yMouse-points_red[1])**2) < 50:
                            drag = True
                            ipx = ix
                            break
                        
                if green_check:
                    xMouse = event.pos[0]
                    yMouse = event.pos[1]
                    # print(xMouse,yMouse)
                    for ix, points_green in enumerate(green_control):
                        if ((xMouse-points_green[0])**2 + (yMouse-points_green[1])**2) < 50:
                            drag = True
                            ipx = ix
                            break
                if blue_check:
                    xMouse = event.pos[0]
                    yMouse = event.pos[1]
                    # print(xMouse,yMouse)
                    for ix, points_blue in enumerate(blue_control):
                        if ((xMouse-points_blue[0])**2 + (yMouse-points_blue[1])**2) < 50:
                            drag = True
                            ipx = ix
                            break
        
                            
        if event.type == pygame.MOUSEBUTTONUP:
            if output_exist:
                if red_check and drag:
                    xMouse = event.pos[0]
                    yMouse = event.pos[1]
                    
                    if ((xMouse-red_control[ipx][0])**2 + (yMouse-red_control[ipx][1])**2) >= 50:
                        # print(red_control[ipx])
                        red_control[ipx] = (xMouse,yMouse)
                        
                        k0 = np.array(red_control)

                        p_red = np.polyfit((k0[:,0]-100)/260,(980-k0[:,1])/260,3)
                        # red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                        
                        # print(p_red)
                        
                        p_red = p_red[3] + p_red[2]*idxx +  p_red[1]*idxx*idxx + p_red[0]*idxx*idxx*idxx
                        red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                        with torch.no_grad():
                            inter_composite, output_composite, par2,= Model.forward_input(
                                torch_bg[None, ...],
                                torch_composite[None, ...],
                                torch_mask[None, ...],
                                [p_red,p_green,p_blue],
                                scale_sm
                            )
                        
                        output_lr = T.ToPILImage()(output_composite[0, ...])
                        inter_lr = T.ToPILImage()(inter_composite[0, ...])

                        composite_lr = T.ToPILImage()(torch_composite)
                        image_gainmap = T.ToPILImage()(
                            (
                                Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                            ).repeat(3, 1, 1)
                        ).resize((256, 256))

                        curves = par2.cpu().detach().numpy()

                        red_curve = curves[0, 0, 0, 0, :]
                        green_curve = curves[0, 1, 0, :, 0]
                        blue_curve = curves[0, 2, :, 0, 0]

                        plt.figure()
                        plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                        plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                        plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                        plt.ylim(0, 1)
                        plt.legend(["Reg", "Green", "Blue"])
                        plt.title("Learned Color Curves")

                        plt.savefig("tmp.jpg")
                        plt.close()

                        curveplot = pygame.image.load("tmp.jpg")
                        curveplot = pygame.transform.scale(curveplot, (256, 256))
                        
                        
                        
                    drag = False
                if green_check and drag:
                    xMouse = event.pos[0]
                    yMouse = event.pos[1]
                    
                    if ((xMouse-green_control[ipx][0])**2 + (yMouse-green_control[ipx][1])**2) >= 50:
                        # print(red_control[ipx])
                        green_control[ipx] = (xMouse,yMouse)
                        
                        k0 = np.array(green_control)

                        p_green = np.polyfit((k0[:,0]-100)/260,(980-k0[:,1])/260,3)
                        # red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                        
                        # print(p_red)
                        
                        p_green = p_green[3] + p_green[2]*idxx +  p_green[1]*idxx*idxx + p_green[0]*idxx*idxx*idxx
                        green_curve_plot = [(100 + ((i/32)*260), 980 - (p_green[i]*260)) for i in range(32)]
                        with torch.no_grad():
                            inter_composite, output_composite, par2,= Model.forward_input(
                                torch_bg[None, ...],
                                torch_composite[None, ...],
                                torch_mask[None, ...],
                                [p_red,p_green,p_blue],
                                scale_sm
                            )
                        
                        output_lr = T.ToPILImage()(output_composite[0, ...])
                        inter_lr = T.ToPILImage()(inter_composite[0, ...])

                        composite_lr = T.ToPILImage()(torch_composite)
                        image_gainmap = T.ToPILImage()(
                            (
                                Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                            ).repeat(3, 1, 1)
                        ).resize((256, 256))

                        curves = par2.cpu().detach().numpy()

                        red_curve = curves[0, 0, 0, 0, :]
                        green_curve = curves[0, 1, 0, :, 0]
                        blue_curve = curves[0, 2, :, 0, 0]

                        plt.figure()
                        plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                        plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                        plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                        plt.ylim(0, 1)
                        plt.legend(["Reg", "Green", "Blue"])
                        plt.title("Learned Color Curves")

                        plt.savefig("tmp.jpg")
                        plt.close()

                        curveplot = pygame.image.load("tmp.jpg")
                        curveplot = pygame.transform.scale(curveplot, (256, 256))
                        
                        
                        
                    drag = False
                if blue_check and drag:
                    xMouse = event.pos[0]
                    yMouse = event.pos[1]
                    
                    if ((xMouse-blue_control[ipx][0])**2 + (yMouse-blue_control[ipx][1])**2) >= 50:
                        # print(red_control[ipx])
                        blue_control[ipx] = (xMouse,yMouse)
                        
                        k0 = np.array(blue_control)

                        p_blue = np.polyfit((k0[:,0]-100)/260,(980-k0[:,1])/260,3)
                        # red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                        
                        # print(p_red)
                        
                        p_blue = p_blue[3] + p_blue[2]*idxx +  p_blue[1]*idxx*idxx + p_blue[0]*idxx*idxx*idxx
                        blue_curve_plot = [(100 + ((i/32)*260), 980 - (p_blue[i]*260)) for i in range(32)]
                        with torch.no_grad():
                            inter_composite, output_composite, par2,= Model.forward_input(
                                torch_bg[None, ...],
                                torch_composite[None, ...],
                                torch_mask[None, ...],
                                [p_red,p_green,p_blue],
                                scale_sm
                            )
                        
                        output_lr = T.ToPILImage()(output_composite[0, ...])
                        inter_lr = T.ToPILImage()(inter_composite[0, ...])

                        composite_lr = T.ToPILImage()(torch_composite)
                        image_gainmap = T.ToPILImage()(
                            (
                                Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                            ).repeat(3, 1, 1)
                        ).resize((256, 256))

                        curves = par2.cpu().detach().numpy()

                        red_curve = curves[0, 0, 0, 0, :]
                        green_curve = curves[0, 1, 0, :, 0]
                        blue_curve = curves[0, 2, :, 0, 0]

                        plt.figure()
                        plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                        plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                        plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                        plt.ylim(0, 1)
                        plt.legend(["Reg", "Green", "Blue"])
                        plt.title("Learned Color Curves")

                        plt.savefig("tmp.jpg")
                        plt.close()

                        curveplot = pygame.image.load("tmp.jpg")
                        curveplot = pygame.transform.scale(curveplot, (256, 256))
                        
                        
                        
                    drag = False
            
        if event.type == pygame.KEYDOWN:
            if output_exist:
                if event.key == pygame.K_a:
                    scale_sm -= 0.03
                    with torch.no_grad():
                            inter_composite, output_composite, par2,= Model.forward_input(
                                torch_bg[None, ...],
                                torch_composite[None, ...],
                                torch_mask[None, ...],
                                [p_red,p_green,p_blue],
                                scale_sm
                            )
                        
                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            ((Model.output_final[0, ...].cpu()-0.6)*(1-scale_sm)/(0.4)+scale_sm) * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))
                    
                 
                if event.key == pygame.K_s:
                    save_ind += 1
                    bg_name = str(file_selection.current_file_path).split("/")[-1].split('.')[0]
                    fg_name = str(fore_selection.current_file_path).split("/")[-1].split('.')[0]
                    print("size:",size_carimg)
                    compite_save = T.ToPILImage()(torch_composite).resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    # compite_save
                    compite_save.save("results_moving_position/%s_%s_%d_composite.jpg"%(bg_name,fg_name,save_ind))
                    
                    
                    mask_save = T.ToPILImage()(torch_mask).resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    # mask_save.resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    mask_save.save("results_moving_position/%s_%s_%d_mask.jpg"%(bg_name,fg_name,save_ind))
                 
                 
                    bg_save = T.ToPILImage()(torch_bg).resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    # bg_save.resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    bg_save.save("results_moving_position/%s_%s_%d_bg.jpg"%(bg_name,fg_name,save_ind))  
                    

                    output_save = T.ToPILImage()(output_composite[0,...]).resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    # output_save.resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    output_save.save("results_moving_position/%s_%s_%d_results.jpg"%(bg_name,fg_name,save_ind))  
                    
                                       
                          
                          
                          
                    plt.figure()
                    plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                    plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                    plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                    plt.ylim(0, 1)
                    plt.legend(["Reg", "Green", "Blue"])
                    plt.title("Learned Color Curves")

                    plt.savefig("results_moving_position/%s_%s_%d_colorcurves.jpg"%(bg_name,fg_name,save_ind))
                    plt.close()         
                    
                    image_gainmap = T.ToPILImage()(
                        (
                            ((Model.output_final[0, ...].cpu()-0.6)*(1-scale_sm)/(0.4)+scale_sm) * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((512, 512)).resize((512,int(512*size_carimg[1]/size_carimg[0])))
                    image_gainmap.save("results_moving_position/%s_%s_%d_gainmap.jpg"%(bg_name,fg_name,save_ind))  
                    image_gainmap = image_gainmap.resize((256,256))
                    
                    
                    
                if event.key == pygame.K_d:
                    scale_sm += 0.03
                    with torch.no_grad():
                            inter_composite, output_composite, par2,= Model.forward_input(
                                torch_bg[None, ...],
                                torch_composite[None, ...],
                                torch_mask[None, ...],
                                [p_red,p_green,p_blue],
                                scale_sm
                            )
                        
                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            ((Model.output_final[0, ...].cpu()-0.6)*(1-scale_sm)/(0.4)+scale_sm) * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))
                
                if event.key == pygame.K_LEFT:
                    translate[0] -= 25

                    torch_bg = transform(image_bg)
                    torch_composite = transform(image_composite)
                    torch_mask = transforms_mask(image_mask)

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=[0, 0], scale=scale, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=[0, 0], scale=scale, shear=0
                    )

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=translate, scale=1, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=translate, scale=1, shear=0
                    )

                    torch_composite = (
                        torch_composite * torch_mask + (1 - torch_mask) * torch_bg
                    )

                    composite_lr = T.ToPILImage()(torch_composite)
                    with torch.no_grad():
                        inter_composite, output_composite, par1, par2 = Model(
                            torch_bg[None, ...],
                            torch_composite[None, ...],
                            torch_mask[None, ...],
                        )

                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))

                    curves = par2.cpu().detach().numpy()

                    red_curve = curves[0, 0, 0, 0, :]
                    green_curve = curves[0, 1, 0, :, 0]
                    blue_curve = curves[0, 2, :, 0, 0]

                    plt.figure()
                    plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                    plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                    plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                    plt.ylim(0, 1)
                    plt.legend(["Reg", "Green", "Blue"])
                    plt.title("Learned Color Curves")

                    plt.savefig("tmp.jpg")
                    plt.close()

                    curveplot = pygame.image.load("tmp.jpg")
                    curveplot = pygame.transform.scale(curveplot, (256, 256))
                    # print("what")
                    # print(pygame.key.get_repeat())
                    idxx = np.linspace(0, 1, 32)
                    p_red = np.polyfit(idxx,red_curve,3)
                    p_red = p_red[3] + p_red[2]*idxx +  p_red[1]*idxx*idxx + p_red[0]*idxx*idxx*idxx
                    
                    p_blue = np.polyfit(idxx,blue_curve,3)
                    p_blue = p_blue[3] + p_blue[2]*idxx +  p_blue[1]*idxx*idxx + p_blue[0]*idxx*idxx*idxx
                    
                    
                    p_green = np.polyfit(idxx,green_curve,3)
                    p_green = p_green[3] + p_green[2]*idxx +  p_green[1]*idxx*idxx + p_green[0]*idxx*idxx*idxx
                    
                    # print(p_red)
                    red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                    # print(red_curve_plot)
                    green_curve_plot = [(100 + (i/32)*260, 980 - p_blue[i]*260) for i in range(32)]
                    blue_curve_plot = [(100 + (i/32)*260, 980 - p_green[i]*260) for i in range(32)]
                    
                    
                    red_control = [red_curve_plot[0],red_curve_plot[10],red_curve_plot[20],red_curve_plot[31]]
                    green_control = [green_curve_plot[0],green_curve_plot[10],green_curve_plot[20],green_curve_plot[31]]
                    blue_control = [blue_curve_plot[0],blue_curve_plot[10],blue_curve_plot[20],blue_curve_plot[31]]
                    break

                if event.key == pygame.K_RIGHT:
                    translate[0] += 25

                    torch_bg = transform(image_bg).to(device)
                    torch_composite = transform(image_composite).to(device)
                    torch_mask = transforms_mask(image_mask).to(device)

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=[0, 0], scale=scale, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=[0, 0], scale=scale, shear=0
                    )

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=translate, scale=1, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=translate, scale=1, shear=0
                    )

                    torch_composite = (
                        torch_composite * torch_mask + (1 - torch_mask) * torch_bg
                    )

                    composite_lr = T.ToPILImage()(torch_composite)
                    with torch.no_grad():
                        inter_composite, output_composite, par1, par2 = Model(
                            torch_bg[None, ...],
                            torch_composite[None, ...],
                            torch_mask[None, ...],
                        )
                    print("b")

                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))

                    curves = par2.cpu().detach().numpy()

                    red_curve = curves[0, 0, 0, 0, :]
                    green_curve = curves[0, 1, 0, :, 0]
                    blue_curve = curves[0, 2, :, 0, 0]

                    plt.figure()
                    plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                    plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                    plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                    plt.ylim(0, 1)
                    plt.legend(["Reg", "Green", "Blue"])
                    plt.title("Learned Color Curves")

                    plt.savefig("tmp.jpg")
                    plt.close()

                    curveplot = pygame.image.load("tmp.jpg")
                    curveplot = pygame.transform.scale(curveplot, (256, 256))
                    print("ca")
                    
                    
                    idxx = np.linspace(0, 1, 32)
                    p_red = np.polyfit(idxx,red_curve,3)
                    p_red = p_red[3] + p_red[2]*idxx +  p_red[1]*idxx*idxx + p_red[0]*idxx*idxx*idxx
                    
                    p_blue = np.polyfit(idxx,blue_curve,3)
                    p_blue = p_blue[3] + p_blue[2]*idxx +  p_blue[1]*idxx*idxx + p_blue[0]*idxx*idxx*idxx
                    
                    
                    p_green = np.polyfit(idxx,green_curve,3)
                    p_green = p_green[3] + p_green[2]*idxx +  p_green[1]*idxx*idxx + p_green[0]*idxx*idxx*idxx
                    
                    # print(p_red)
                    red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                    # print(red_curve_plot)
                    green_curve_plot = [(100 + (i/32)*260, 980 - p_blue[i]*260) for i in range(32)]
                    blue_curve_plot = [(100 + (i/32)*260, 980 - p_green[i]*260) for i in range(32)]
                                           
                    red_control = [red_curve_plot[0],red_curve_plot[10],red_curve_plot[20],red_curve_plot[31]]
                    green_control = [green_curve_plot[0],green_curve_plot[10],green_curve_plot[20],green_curve_plot[31]]
                    blue_control = [blue_curve_plot[0],blue_curve_plot[10],blue_curve_plot[20],blue_curve_plot[31]]
                    
                    
                    break
                if event.key == pygame.K_UP:
                    translate[1] -= 25

                    torch_bg = transform(image_bg).to(device)
                    torch_composite = transform(image_composite).to(device)
                    torch_mask = transforms_mask(image_mask).to(device)

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=[0, 0], scale=scale, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=[0, 0], scale=scale, shear=0
                    )

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=translate, scale=1, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=translate, scale=1, shear=0
                    )

                    torch_composite = (
                        torch_composite * torch_mask + (1 - torch_mask) * torch_bg
                    )

                    composite_lr = T.ToPILImage()(torch_composite)

                    with torch.no_grad():
                        inter_composite, output_composite, par1, par2 = Model(
                            torch_bg[None, ...],
                            torch_composite[None, ...],
                            torch_mask[None, ...],
                        )

                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))

                    curves = par2.cpu().detach().numpy()

                    red_curve = curves[0, 0, 0, 0, :]
                    green_curve = curves[0, 1, 0, :, 0]
                    blue_curve = curves[0, 2, :, 0, 0]

                    plt.figure()
                    plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                    plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                    plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                    plt.ylim(0, 1)
                    plt.legend(["Reg", "Green", "Blue"])
                    plt.title("Learned Color Curves")

                    plt.savefig("tmp.jpg")
                    plt.close()

                    curveplot = pygame.image.load("tmp.jpg")
                    curveplot = pygame.transform.scale(curveplot, (256, 256))
                    idxx = np.linspace(0, 1, 32)
                    p_red = np.polyfit(idxx,red_curve,3)
                    p_red = p_red[3] + p_red[2]*idxx +  p_red[1]*idxx*idxx + p_red[0]*idxx*idxx*idxx
                    
                    p_blue = np.polyfit(idxx,blue_curve,3)
                    p_blue = p_blue[3] + p_blue[2]*idxx +  p_blue[1]*idxx*idxx + p_blue[0]*idxx*idxx*idxx
                    
                    
                    p_green = np.polyfit(idxx,green_curve,3)
                    p_green = p_green[3] + p_green[2]*idxx +  p_green[1]*idxx*idxx + p_green[0]*idxx*idxx*idxx
                    
                    # print(p_red)
                    red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                    # print(red_curve_plot)
                    green_curve_plot = [(100 + (i/32)*260, 980 - p_blue[i]*260) for i in range(32)]
                    blue_curve_plot = [(100 + (i/32)*260, 980 - p_green[i]*260) for i in range(32)]
                    
                    
                    red_control = [red_curve_plot[0],red_curve_plot[10],red_curve_plot[20],red_curve_plot[31]]
                    green_control = [green_curve_plot[0],green_curve_plot[10],green_curve_plot[20],green_curve_plot[31]]
                    blue_control = [blue_curve_plot[0],blue_curve_plot[10],blue_curve_plot[20],blue_curve_plot[31]]
                    break

                if event.key == pygame.K_DOWN:
                    translate[1] += 25

                    torch_bg = transform(image_bg).to(device)
                    torch_composite = transform(image_composite).to(device)
                    torch_mask = transforms_mask(image_mask).to(device)

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=[0, 0], scale=scale, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=[0, 0], scale=scale, shear=0
                    )

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=translate, scale=1, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=translate, scale=1, shear=0
                    )

                    torch_composite = (
                        torch_composite * torch_mask + (1 - torch_mask) * torch_bg
                    )

                    composite_lr = T.ToPILImage()(torch_composite)

                    with torch.no_grad():
                        inter_composite, output_composite, par1, par2 = Model(
                            torch_bg[None, ...],
                            torch_composite[None, ...],
                            torch_mask[None, ...],
                        )

                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))

                    curves = par2.cpu().detach().numpy()

                    red_curve = curves[0, 0, 0, 0, :]
                    green_curve = curves[0, 1, 0, :, 0]
                    blue_curve = curves[0, 2, :, 0, 0]

                    plt.figure()
                    plt.plot(np.linspace(0, 1, 32), red_curve, "r")
                    plt.plot(np.linspace(0, 1, 32), green_curve, "g")
                    plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
                    plt.ylim(0, 1)
                    plt.legend(["Reg", "Green", "Blue"])
                    plt.title("Learned Color Curves")

                    plt.savefig("tmp.jpg")
                    plt.close()

                    curveplot = pygame.image.load("tmp.jpg")
                    curveplot = pygame.transform.scale(curveplot, (256, 256))
                    idxx = np.linspace(0, 1, 32)
                    p_red = np.polyfit(idxx,red_curve,3)
                    p_red = p_red[3] + p_red[2]*idxx +  p_red[1]*idxx*idxx + p_red[0]*idxx*idxx*idxx
                    
                    p_blue = np.polyfit(idxx,blue_curve,3)
                    p_blue = p_blue[3] + p_blue[2]*idxx +  p_blue[1]*idxx*idxx + p_blue[0]*idxx*idxx*idxx
                    
                    
                    p_green = np.polyfit(idxx,green_curve,3)
                    p_green = p_green[3] + p_green[2]*idxx +  p_green[1]*idxx*idxx + p_green[0]*idxx*idxx*idxx
                    
                    # print(p_red)
                    red_curve_plot = [(100 + ((i/32)*260), 980 - (p_red[i]*260)) for i in range(32)]
                    green_curve_plot = [(100 + (i/32)*260, 980 - p_blue[i]*260) for i in range(32)]
                    blue_curve_plot = [(100 + (i/32)*260, 980 - p_green[i]*260) for i in range(32)]
                    
                    
                    red_control = [red_curve_plot[0],red_curve_plot[10],red_curve_plot[20],red_curve_plot[31]]
                    green_control = [green_curve_plot[0],green_curve_plot[10],green_curve_plot[20],green_curve_plot[31]]
                    blue_control = [blue_curve_plot[0],blue_curve_plot[10],blue_curve_plot[20],blue_curve_plot[31]]
                    break

                if event.key == pygame.K_SPACE:
                    torch_bg = transform(image_bg).to(device)
                    torch_composite = transform(image_composite).to(device)
                    torch_mask = transforms_mask(image_mask).to(device)
                    print(b)
                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=[0, 0], scale=scale, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=[0, 0], scale=scale, shear=0
                    )

                    torch_composite = F.affine(
                        torch_composite, angle=0, translate=translate, scale=1, shear=0
                    )
                    torch_mask = F.affine(
                        torch_mask, angle=0, translate=translate, scale=1, shear=0
                    )

                    torch_composite = (
                        torch_composite * torch_mask + (1 - torch_mask) * torch_bg
                    )

                    with torch.no_grad():
                        inter_composite, output_composite, par1, par2 = Model(
                            torch_bg[None, ...],
                            torch_composite[None, ...],
                            torch_mask[None, ...],
                        )

                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    composite_lr = T.ToPILImage()(torch_composite)
                    inter_lr = T.ToPILImage()(inter_composite[0, ...])

                    composite_lr = T.ToPILImage()(torch_composite)
                    image_gainmap = T.ToPILImage()(
                        (
                            Model.output_final[0, ...].cpu() * torch_mask[0, ...].cpu()
                        ).repeat(3, 1, 1)
                    ).resize((256, 256))
                    # print(output_lr)
                    curves = par2.cpu().detach().numpy()

                    red_curve = curves[0, 0, 0, 0, :]
                    green_curve = curves[0, 1, 0, :, 0]
                    blue_curve = curves[0, 2, :, 0, 0]

                    plt.figure()
                    plt.plot(red_curve, "r")
                    plt.plot(green_curve, "g")
                    plt.plot(blue_curve, "b")
                    plt.ylim(0, 1)
                    plt.legend(["Reg", "Green", "Blue"])
                    plt.title("Learned Color Curves")

                    plt.savefig("tmp.jpg")
                    plt.close()

                    curveplot = pygame.image.load("tmp.jpg")
                    curveplot = pygame.transform.scale(curveplot, (256, 256))
                    break

        manager.process_events(event)

    if (image_selected or model_selected) or fore_selected:
        window_surface.blit(background, (0, 0))

    if image_selected:

        window_surface.blit(carImg, (0, 180))

    if output_exist:
        if not flag_dialog:
            window_surface.blit(pilImageToSurface(composite_lr), (0, 180))
        window_surface.blit(pilImageToSurface(output_lr), (1160, 180))
        window_surface.blit(curveplot, (600, 700))
        window_surface.blit(pilImageToSurface(image_gainmap), (1200, 700))


        window_surface.blit(pilImageToSurface(inter_lr), (580, 180))
        pygame.draw.rect(window_surface, (255,255,255), pygame.Rect(80, 700, 300, 300),2)
        pygame.draw.lines(window_surface, (255,0,0), False, [(x[0], x[1]) for x in red_curve_plot],width=2)
        pygame.draw.lines(window_surface, (0,255,0), False, [(x[0], x[1]) for x in green_curve_plot],width=2)
        pygame.draw.lines(window_surface, (0,0,255), False, [(x[0], x[1]) for x in blue_curve_plot],width=2)
        
        if red_check:
            for points in red_control:
                pygame.draw.circle(window_surface,(255,0,0),points,5)
        if green_check:
            for points in green_control:
                pygame.draw.circle(window_surface,(0,255,0),points,5)
        
        if blue_check:
            for points in blue_control:
                pygame.draw.circle(window_surface,(0,0,255),points,5)
    window_surface.blit(text, textRect)
    window_surface.blit(text_fore, textRect_fore)

    window_surface.blit(text_model, textRect_model)

    manager.update(time_delta)
    manager.draw_ui(window_surface)

    # pygame.draw.lines(window_surface, (255,255,0), False, [(100,200),(200,300),(300,400)])
    pygame.display.update()
