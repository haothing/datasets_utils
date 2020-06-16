import os
import sys, sqlite3
import copy
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.io

# command sample
# synth_text.py -n 200 -b "E:/datasets/SynthText/bg_images" -o "E:/datasets/SynthText/SynthText_Gen/SynthText_self" -ogt "E:/datasets/SynthText/SynthText_Gen" -mat 
# synth_text.py -n 10 -f ./fonts/jp -o "E:/datasets/SynthText/test/images" -b "E:/datasets/SynthText/bg_images_1" -ogt "E:/datasets/SynthText/test/test.mat" -mat
# synth_text.py -n 2 -f ./fonts/jp -o "/home/repo/datasets/SynthText/test/images" -b "./bg_images" -ogt "/home/repo/datasets/SynthText/test/test.mat" -mat
# synth_text.py -n 10 -o "E:/datasets/SynthText/test/images" -ogt "E:/datasets/SynthText/test/" -l
# synth_text.py -n 1000 -cn 12 -c ./characters/japanese_kana.txt -f ./fonts/jp -o "E:/datasets/SynthText/ja_kana/images" -ogt "E:/datasets/SynthText/ja_kana/ground_true.txt" -l

parser = argparse.ArgumentParser(description='Text Label Image Generator')
parser.add_argument('--background_folder', '-b', default='./bg_images', type=str, help='images folder as background')
parser.add_argument('--result_num', '-n', default=100, type=int, help='number of result images every background image, as total when no background images.')
parser.add_argument('--characters_num', '-cn', default=5, type=int, help='number of characters in image.')
parser.add_argument('--output_images_folder', '-o', default='./', type=str, help='output images folder')
parser.add_argument('--output_ground_true', '-ogt', default='./', type=str, help='output ground true file')
parser.add_argument('--mat_ground_true', '-mat', default=False, action='store_true', help='output mat ground true')
parser.add_argument('--one_line', '-l', default=False, action='store_true', help='generate one line text images')
parser.add_argument('--fonts_folder', '-f', default='./fonts/jp', type=str, help='folder of fonts that to draw text')
parser.add_argument('--characters_file', '-c', default='./characters/japanese_char.txt', type=str, help='origin characters to make image files')

parser.add_argument('--height', default=32, type=int, help='number of result images height')
parser.add_argument('--width', default=280, type=int, help='number of result images width')
# parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
# parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
# parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
# parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
# parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
# parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
# parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
# parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
# parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

def text_generator(result_total=10):
    
    num_text_list = np.random.normal(loc=10, scale=3, size=result_total).astype(int)
    num_text_list[num_text_list <= 0] = 1
    
    # 随机得到日语短句
    conn = sqlite3.connect("./wnjpn.db")
    cur = conn.execute("select * from word where lang = 'jpn' ORDER BY RANDOM() LIMIT " + str(np.sum(num_text_list)) + ";")
    word_list = [record[2] for record in cur.fetchall()]

    mark_num = np.random.randint(0, 5, size=np.sum(num_text_list))
    mark = '123456789012345678901234567890-^\@[]:;,./\!"#$%&\'()=~|{`+*}_?><１２３４５６７８９０１２３４５６７８９０１２３４５６７８９０ー＾￥「＠」：；、。・￥！”＃＄％＆’（）＝～｜｛‘＋｝＊＿？＞＜'
    tail = [''.join([mark[np.random.randint(len(mark))] for add_c in range(mark_num[i])]) for i in range(len(mark_num))]

    word_list = [word_list[i] + tail[i] for i in range(len(word_list))]

    return num_text_list, word_list

def image_generator():

    # return list object with image file name and word bound boxes and char bound boxes and text 
    # imnames, wordBB, charBB, txt

    # SynthText 格式
    bg_path = args.background_folder
    output_path = args.output_images_folder
    images_folder = os.path.split(output_path)[1]
    files= os.listdir(bg_path)
    y_margin, total = 30, args.result_num
    #imnames, wordBB, charBB, txt = [], np.zeros(total).astype(object), np.zeros(total).astype(object), []
    imnames, wordBB, charBB, txt, font_list = [], [], [], [], []

    for file_name in os.listdir(args.fonts_folder):
        font_list.append(os.path.join(args.fonts_folder, file_name))

    for file_index, file_name in enumerate(files):

        if os.path.splitext(file_name)[1].lower() in ('.jpg', '.jpeg'):
            file_path = os.path.join(bg_path, file_name)

        word_index = 0
        num_text_list, word_list = text_generator(result_total=total)
        for index in range(total):

            bg_img = Image.open(file_path, mode="r")
            bk_np = np.asarray(bg_img)
            draw = ImageDraw.Draw(bg_img)

            num_text = num_text_list[index]
            if num_text <= 0:
                continue
            size = np.random.randint(30, 80, size=num_text)

            # 减少文本超越图片大小的概率，缩小取得位置的起始点
            pos_x = np.random.randint(bk_np.shape[1] * 0.7, size=num_text)
            pos_y = np.arange(num_text) * bk_np.shape[0] / num_text
            pos_xy = np.stack((pos_x, pos_y), 1)

            wordBB_item, charBB_item, txt_item = [], [], []
            for i in range(num_text):

                wh = np.asarray((bg_img.width, bg_img.height))

                fnt_path = font_list[np.random.randint(0, len(font_list))]
                fnt = ImageFont.truetype(fnt_path, size[i])
                text = word_list[word_index]

                #print(bk_np.shape, tuple(np.flip(pos_xy[i])))
                rgb = bk_np[tuple(np.flip(pos_xy[i]).astype(np.int32))] + 100
                near_rbg = np.where(rgb > 255, rgb - 255, rgb)

                # compute text size with characters space and characters
                xy = pos_xy[i].copy()
                textsize_w, textsize_h = 0, 0
                for c_i in range(len(text)):
                    c = text[c_i]
                    draw.text(xy, c, font=fnt, fill=tuple(near_rbg))
                    
                    c_size = draw.textsize(c, font=fnt)
                    charBB_item.append(copy.deepcopy([xy, xy + (c_size[0], 0), xy + c_size, xy + (0, c_size[1])]))
                    
                    xy += (c_size[0], 0)
                    textsize_w += c_size[0]
                    textsize_h = c_size[1] if c_size[1] > textsize_h else textsize_h

                textsize = [textsize_w, textsize_h]

                # 超出图片范围时跳过
                #if (wh - pos_xy[i] - textsize).min() < 0:
                #    continue
                #rect_xy = np.hstack((pos_xy[i] + (-1, 1), pos_xy[i] + (-1, 1) + textsize))
                #draw.rectangle(rect_xy.tolist(), outline="green", width=1)

                wordBB_item.append([pos_xy[i], pos_xy[i] + (textsize[0], 0), pos_xy[i] + textsize, pos_xy[i] + (0, textsize[1])])
                txt_item.append(text)
                
                word_index += 1
                #xywh = np.hstack((np.array(0.), np.asarray(pos_xy[i] + textsize / 2) / wh, textsize / wh))
                #bboxes_list.append(xywh)

            #result_path = os.path.join(file_name.split('.')[0], str(index) + '.jpg')
            new_file_name = str(file_index * total + index) + '.jpg'
            #full_path = os.path.join(output_path, 'SynthText_self')
            os.makedirs(output_path, exist_ok=True)
            bg_img.save(os.path.join(output_path, new_file_name))

            wordBB.append(np.array(wordBB_item, dtype=np.float64))
            charBB.append(np.array(charBB_item, dtype=np.float64))
            txt.append(np.array(txt_item))
            imnames.append(np.array([os.path.join(images_folder, new_file_name)]))

        print('file_name:{}, generate to {:d} files.'.format(file_name, index + 1))

    return imnames, wordBB, charBB, txt


def get_text(char_num):
    
    args.characters_file

    char_file = open(args.characters_file, "r", encoding="utf-8")
    #char_file = open('./char_test.txt', "r", encoding="utf-8")
    all_text = ''.join(char_file.read().splitlines())
    text = ''.join([all_text[np.random.randint(len(all_text))] for i in range(char_num)])

    return text

def one_line_images():

    # return list object with image file name and text 
    os.makedirs(args.output_images_folder, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_ground_true), exist_ok=True)

    output_path = args.output_images_folder
    images_folder = os.path.split(output_path)[1]
    total = args.result_num
    
    font_list, gt_text = [], []
    for file_name in os.listdir(args.fonts_folder):
        font_list.append(os.path.join(args.fonts_folder, file_name))

    for index in range(total):
        bg_color = np.random.randint(0, 255)
        text_color = bg_color + 127 - 255 if bg_color + 127 > 255 else bg_color + 127
        
        img = np.full((args.height, args.width), bg_color, dtype=np.uint8)
        img = Image.fromarray(img, mode='L')

        fnt_path = font_list[np.random.randint(0, len(font_list))]
        fnt = ImageFont.truetype(fnt_path, int(args.height * (np.random.randint(6, 10) / 10)))
        text = get_text(args.characters_num)

        draw = ImageDraw.Draw(img)
        draw_w = draw.textsize(text, font=fnt)[0]
        if draw_w > args.width:
            text = text[:int(args.width / draw_w * len(text))]
        draw.text((0, 0), text, font=fnt, fill=text_color)

        img_path = os.path.join(output_path, str(index) + '.jpg')
        img.save(img_path)
        gt_text.append(os.path.join(images_folder, str(index) + '.jpg') + ' ' + text)

        print('\rgenerated files {:d}/{:d}'.format(index + 1, total), end='')

    gt_file = open(args.output_ground_true, "w", encoding="utf-8")
    gt_file.write("\n".join(gt_text))
    gt_file.close()

    print('')

def save_mat_gt(imnames, wordBB, charBB, txt):

    wordBB, charBB = np.array(wordBB), np.array(charBB)
    for i in range(len(wordBB)):
        wordBB[i] = wordBB[i].T

    for i in range(len(charBB)):
        charBB[i] = charBB[i].T

    scipy.io.savemat(args.output_ground_true, 
        {'imnames': np.array([imnames]), 'wordBB': np.array([wordBB]), 'charBB': np.array([charBB]), 'txt': np.array([txt])})

def save_simple_gt(imnames, txt):
    # save result as simple ground true
    # 20456343_4045240981.jpg your text
    # 20457281_3395886438.jpg second line

    f = open(args.output_ground_true, "w", encoding="utf-8")
    gt_text = []
    for i, file_name in enumerate(imnames):
        gt_text.append(file_name[0] + ' ' + ','.join(txt[i]))
    f.write("\n".join(gt_text))
    f.close()

if __name__ == '__main__':
    
    if args.one_line:
        one_line_images()
    else:
        imnames, wordBB, charBB, txt = image_generator()
        if args.mat_ground_true:
            save_mat_gt(imnames, wordBB, charBB, txt)
        else:
            save_simple_gt(imnames, txt)
