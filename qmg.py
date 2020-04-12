from sys import argv
from os import listdir, system
from os.path import join, isdir
from html import escape
import re

OUTPUT_DIR = 'E:\\Image\\Manga\\'

r = re.compile(r'.*\[(?P<author>.+)\] (?P<title>.+)$')
meta = '''<?xml version="1.0" encoding="utf-8"?>
<ComicInfo xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <Series>{title}</Series>
    <Writer>{author}</Writer>
</ComicInfo>
'''


def dirtree(base):
    worklist = []
    # 不考虑出错的情况
    for lv1 in base:
        lv2 = [join(lv1, x) for x in listdir(lv1) if x.lower() != 'comicinfo.xml']
        if not any(map(isdir, lv2)):
            worklist.append(lv1)
            continue
        lv3 = (join(dir, x) for dir in lv2 for x in listdir(dir) if x.lower() != 'commicinfo.xml')
        if not any(map(lambda x: isdir(join(lv1, x)), lv3)):
            worklist.append(lv1)
        else:
            worklist += lv2
    return worklist

def convert(src, dst, format):
    match = r.match(escape(src))
    if match:
        metapath = join(src, 'ComicInfo.xml')
        # metadata = {
        #     'author': match.groups()[0],
        #     'title': match.groups()[1]
        # }
        with open(metapath, 'w', encoding='utf-8') as f:
            f.write(meta.format_map(match.groupdict()))
    system(f'kcc-c2e.exe -p KV -f {format.upper()} -u -r 1 -o {dst} --forcecolor "{src}"')

def main(baselist, output=OUTPUT_DIR, format='MOBI'):
    worklist = dirtree(baselist)
    m=len(worklist)
    for i, x in enumerate(worklist):
        print(f'================================== {i+1}/{m} ==================================')
        print(x)
        convert(x, output, format)

if __name__=='__main__':
    main((x for x in (argv if len(argv)>1 else listdir()) if isdir(x)))