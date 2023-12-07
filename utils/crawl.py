import numpy as np
from selenium import webdriver
import time
import os
from enum import Enum
from PIL import Image, ImageFile
import errno
import re
import urllib.request
from multiprocessing import Pool
from tqdm import tqdm

proxy = urllib.request.ProxyHandler({'http': ' '})  # set proxy IP
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', ' ')]  # set user agent
urllib.request.install_opener(opener)

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--headless")

browser = webdriver.Chrome(executable_path='~chromedriver.exe',
                           chrome_options=chrome_options)


class EntityName(Enum):
    YAGO = './Datasets/YAGO15K/'
    DB = './Datasets/DB15K/'
    FB = './Datasets/FB15K/'
    WN = './Datasets/WN18/'


Key = EntityName.WN
text_len = 8
min_crawl_imgs = 10
max_crawl_imgs = 40
multi_process = 8
sleep_time = 3
image_dir = "~\\Datasets\\{}-IMG-GOOGLE".format(Key.name)
url_dir_no_res = './cat/NoneDB.txt'

dict_ent2id = {}
dict_id2text = {}
with open(Key.value + 'entity2id.txt', 'r', encoding='utf-8') as f1:
    lines1 = f1.readlines()
    for ln in lines1:
        if Key.name == 'DB' or Key.name == 'YAGO':
            ent, id = ln.split('resource/')[1].split('> ')
        elif Key.name == 'FB':
            ent, id = ln.split(' ')
        else:
            ent, id = ln.split('\t')
        dict_ent2id[ent] = id.rstrip('\n')

dict_ent2name = {}
if Key.name == 'FB':
    with open(Key.value + 'entity2text.txt', 'r', encoding='utf-8') as f11:
        lines11 = f11.readlines()
        for ln in lines11:
            ent, name = ln.split('\t')
            dict_ent2name[ent] = name.rstrip('\n')


def remove_parentheses(text):
    stack = []
    result = ''
    for char in text:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                result += char
        elif not stack:
            result += char
    return result


pattern = r"\/.*?\/"
with open(Key.value + 'entity_description.txt', 'r', encoding='utf-8') as f2:
    lines2 = f2.readlines()
    for ln in lines2:
        id, text = ln.split("\t")
        if text == 'No description defined\n' or text == 'Unknown\n':
            text = ""
        else:
            text = remove_parentheses(text)  # remove parentheses and any contents inside
            text = re.sub(pattern, "", text)
            text = re.sub(r'\s+([^\w\s])', r'\1', text)  # remove space before punctuation
            text_pre = re.sub(r'\s+', ' ', text)  # remove extra space
            list0 = text_pre.split(' ')
            if len(list0) < text_len:
                text = text_pre.rstrip('\n')
            else:
                text1 = text_pre.split(".")[0].rstrip('\n')
                list1 = text1.split(' ')
                if len(list1) > text_len:
                    my_list = list1[:text_len]
                    text = ' '.join(my_list).rstrip(',')
                else:
                    text = ' '.join(list0[:text_len]).rstrip(',')

        dict_id2text[id] = text


def download_image(url, count, id):
    freebase_id, index = id, count
    index = int(index)

    # Create dir for entity if it doesn't exist.
    target_dir = os.path.join(image_dir, freebase_id)

    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    except OSError as e:  # multiprocessing-safe way to handle existing dir
        if e.errno != errno.EEXIST:
            raise

    # TODO: Maybe use more images because some cannot be downloaded.
    if index < max_crawl_imgs:  # only use first 50 images (as in the paper)
        target_filename = '{}.jpg'.format(index)

        # Skip images that were already downloaded.
        if os.path.exists(os.path.join(target_dir, target_filename)):
            pass
        else:
            try:
                # Download and open image.
                # TODO: Store temporary files in a separate folder (e.g. image-graph_temp),
                #       so they are not hidden in the system folders.
                temp_filename = ''
                temp_filename, _ = urllib.request.urlretrieve(url)
                im = Image.open(temp_filename)
            # except (IOError, CertificateError, HTTPException):
            #     pass
            except Exception as e:
                # print('Got unusual error during downloading/opening of image:', e)
                with open(url_dir_no_res, 'a+', encoding='utf-8') as f:
                    f.write(url + '\t' + id + '/' + str(index) + '\n')
                # print('Please make sure that this error is just caused by a corrupted file.')
            else:
                # Resize and convert to jpg.
                im.thumbnail((224, 224), Image.ANTIALIAS)
                im = im.convert('RGB')
                im.save(os.path.join(target_dir, target_filename))
            finally:
                # Remove temporary file.
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass


class Crawler_google_images:

    def __init__(self, Key):
        self.Key = Key

    def capture_image_urls(self, url, id):
        picpath = './cat'
        if not os.path.exists(picpath):
            os.makedirs(picpath)

        img_url_dic = []
        count = 0

        browser.get(url)
        browser.maximize_window()
        for i in range(2):
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(sleep_time)

            # img_elements = browser.find_elements_by_id('islmp')[0]
            # img_elements = img_elements.find_elements_by_class_name('bRMDJf')
            img_elements = browser.find_elements_by_css_selector('img.rg_i')

            for img_element in img_elements:
                # img_element = img_element.find_elements_by_tag_name('img')[0]
                img_url = img_element.get_attribute('src')

                if isinstance(img_url, str):
                    if img_url not in img_url_dic:
                        download_image(img_url, count, id)
                        img_url_dic.append(img_url)

                        count += 1
                        if count > max_crawl_imgs:
                            browser.get('https://www.google.com/search')
                            return img_url_dic

        browser.get('https://www.google.com/search')
        return img_url_dic

    def process_task(self, url, id):
        img_url_dic = self.capture_image_urls(url, id)
        if img_url_dic is None or len(img_url_dic) < min_crawl_imgs:
            print(url)
            with open('./cat/{}_URLS_None.txt'.format(Key.name), 'a+', encoding='utf-8') as f:
                f.write(str(url) + '\n')
        else:
            with open('./cat/{}_URLS_id.txt'.format(Key.name), 'a+', encoding='utf-8') as f9:
                f9.write(str(id) + '\n')
            path = './cat/{}_URLS_google.txt'.format(Key.name)
            with open(path, 'a+', encoding='utf-8') as f:
                for i in range(len(img_url_dic)):
                    f.write(img_url_dic[i] + '\t' + id + '/' + str(i) + '\n')

    def run(self):

        with open('./cat/{}_URLS_id.txt'.format(Key.name), 'r', encoding='utf-8') as fi:
            ids = fi.readlines()
            ids = [x.strip('\n') for x in ids]

        # sub_dirs = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
        # result = []
        # for sub_dir in sub_dirs:
        #     sub_dir_path = os.path.join(image_dir, sub_dir)
        #     image_count = len([name for name in os.listdir(sub_dir_path) if name.endswith('.jpg')])
        #     if image_count > 15:
        #         result.append(sub_dir)

        # can = np.zeros(len(dict_ent2id))
        # for ele in tqdm(os.listdir(image_dir)):
        #     can[int(ele)] = 1
        # ones_indices = np.where(can == 1)[0]
        # zeros_indices = np.where(can == 0)[0]
        # print(zeros_indices)

        with Pool(processes=multi_process) as pool:
            with tqdm(total=len(dict_ent2id)) as pbar:
                for ent, id in dict_ent2id.items():

                    if id in ids:
                        print(id)
                        pbar.update()
                        continue

                    # if int(id) in list(ones_indices):
                    #     pbar.update()
                    #     continue

                    # if str(id) in result:
                    #     print(id)
                    #     pbar.update()
                    #     continue

                    if Key.name != 'FB':
                        ent_txt = ent.replace('_', ' ')
                        text = dict_id2text[id]
                    else:
                        ent_txt = dict_ent2name[ent]
                        id = dict_ent2id[ent]
                        text = dict_id2text[id]

                    if text != "" and Key.name == 'WN':
                        final_text = text.split(';')[0]
                        url = 'https://www.google.com/search?q=' + final_text + '&tbm=isch'
                    elif text != "":
                        url = 'https://www.google.com/search?q=' + ent_txt + ', ' + text + '&tbm=isch'
                    else:
                        url = 'https://www.google.com/search?q=' + ent_txt + '&tbm=isch'
                    pool.apply_async(self.process_task, args=(url, id),
                                     callback=lambda x: pbar.update())
                pool.close()
                pool.join()

            # with open('./cat/{}_URLS_None.txt'.format(Key.name), 'r', encoding='utf-8') as f11:
            #     urls_left = f11.readlines()
            # dict_name2ent = {v: k for k, v in dict_ent2name.items()}
            # with Pool(processes=10) as pool:
            #     with tqdm(total=len(urls_left)) as pbar:
            #         for url in urls_left:
            #             ent_text = url.split('https://www.google.com/search?q=')[1].split(', ')[0].rstrip('&tbm=isch')
            #             ent = dict_name2ent[ent_text]
            #             id = dict_ent2id[ent]
            #
            #         pool.apply_async(self.process_task, args=(url, id),
            #                          callback=lambda x: pbar.update())
            #         pool.close()
            #         pool.join()

        browser.close()


if __name__ == '__main__':
    craw = Crawler_google_images(Key)
    craw.run()
