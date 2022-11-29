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
while 1:
    time_delta = clock.tick(60) / 1000.0

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
                        print(file_selection.current_file_path)
                        carImg = pygame.image.load(file_selection.current_file_path)
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
                        print(fore_selection.current_file_path)
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
                        print(model_selection.current_file_path)

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
                        Model = Model_Composite_PL(dim=32)
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
                        composite_lr = T.ToPILImage()(torch_composite)
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
                        curveplot = pygame.transform.scale(curveplot, (512, 512))

                        output_exist = True

        if event.type == pygame.KEYDOWN:
            if output_exist:
                if event.key == pygame.K_LEFT:
                    translate[0] -= 10

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
                    curveplot = pygame.transform.scale(curveplot, (512, 512))

                if event.key == pygame.K_RIGHT:
                    translate[0] += 10

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
                    curveplot = pygame.transform.scale(curveplot, (512, 512))
                if event.key == pygame.K_UP:
                    translate[1] -= 10

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
                    curveplot = pygame.transform.scale(curveplot, (512, 512))
                if event.key == pygame.K_DOWN:
                    translate[1] += 10

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
                    curveplot = pygame.transform.scale(curveplot, (512, 512))

                if event.key == pygame.K_SPACE:
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

                    with torch.no_grad():
                        inter_composite, output_composite, par1, par2 = Model(
                            torch_bg[None, ...],
                            torch_composite[None, ...],
                            torch_mask[None, ...],
                        )

                    output_lr = T.ToPILImage()(output_composite[0, ...])
                    composite_lr = T.ToPILImage()(torch_composite)
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
                    curveplot = pygame.transform.scale(curveplot, (512, 512))
        manager.process_events(event)

    if (image_selected or model_selected) or fore_selected:
        window_surface.blit(background, (0, 0))

    if image_selected:

        window_surface.blit(carImg, (0, 300))

    if output_exist:
        window_surface.blit(pilImageToSurface(composite_lr), (0, 300))
        window_surface.blit(pilImageToSurface(output_lr), (600, 300))
        window_surface.blit(curveplot, (1200, 300))

    window_surface.blit(text, textRect)
    window_surface.blit(text_fore, textRect_fore)

    window_surface.blit(text_model, textRect_model)

    manager.update(time_delta)
    manager.draw_ui(window_surface)

    pygame.display.update()
