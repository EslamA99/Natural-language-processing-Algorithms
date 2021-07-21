import re
from tkinter import *



def readFile(fileName):
    f = open(fileName, encoding='utf-8')
    text = f.read()
    return text


def parseToArr(txt):
    text = re.sub('[^0-9\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc ]', ' ', txt)

    arr = re.split(' |\t', text)
    arr = [x for x in arr if x]
    return arr


def fillChainOfD(words):
    d = {}
    for i in range(len(words) - 1):
        if words[i] in d:
            arr = d[words[i]]
            arr[0] += 1
        else:
            arr = [1, {}]
            d[words[i]] = arr
        if words[i + 1] in arr[1]:
            arr2 = arr[1][words[i + 1]]
            arr2[0] += 1
        else:
            arr2 = [1, {}]
            arr[1][words[i + 1]] = arr2
        if i <= len(words) - 3:
            if words[i + 2] in arr2[1]:
                arr3 = arr2[1][words[i + 2]]
                arr3[0] += 1
            else:
                arr3 = [1, {}]
                arr2[1][words[i + 2]] = arr3

    return d


def autoComplete(d, data):
    words = parseToArr(data)
    out = []
    if len(words) == 0:
        out = []
    elif len(words) == 1:
        for key in sorted(d[words[0]][1], key=lambda name: d[words[0]][1][name][0], reverse=True):
            out.append(data.strip() + ' ' + key)
        count = d[words[0]][0]
        for key, value in d[words[0]][1].items():
            print(data .strip()+ ' ' + key, float(value[0]) / float(count))
    else:
        prob = 1.0
        for i in range(len(words) - 1):
            firstWord = words[i]
            secondWord = words[i + 1]
            if firstWord in d:
                if secondWord in d[firstWord][1]:
                    d3 = d[firstWord][1][secondWord][1]
                    bigramCount = d[firstWord][0]
                else:
                    prob = 0.0
                    break
            else:
                prob = 0.0
                break
            if i <= len(words) - 3:
                thirdWord = words[i + 2]
                # print(d3)
                if thirdWord in d3:
                    tigramCount = d3[thirdWord][0]
                    prob *= tigramCount / float(bigramCount)
                else:
                    prob = 0.0
                    break
        if prob != 0.0:
            tmp = d[words[len(words) - 2]][1][words[len(words) - 1]]
            for key, value in tmp[1].items():
                print(data.strip() + ' ' + key, prob * float(value[0]) / float(tmp[0]))
            out = []
            for key in sorted(tmp[1], key=lambda name: tmp[1][name][0], reverse=True):
                out.append(data.strip() + ' ' + key)
    return out


def clearAndUpdate(data):
    result_list.delete(0, END)
    for item in data:
        result_list.insert(END, item)


def checkSearch(_):
    typed = search_entry.get()
    if len(typed) != 0:
        clearAndUpdate(autoComplete(d, typed))


def getD():
    txt = readFile('dataTemp.txt')
    arr = parseToArr(txt)
    d = fillChainOfD(arr)
    return d


d = getD()
root = Tk()
root.geometry("500x600")

label_search = Label(root, text="Search for a word: ", )
label_search.pack(pady=20)

search_entry = Entry(root, width=70, )
search_entry.pack()

result_list = Listbox(root, width=70, height=200)
result_list.pack(pady=40, )
search_entry.bind("<KeyRelease>", checkSearch)
root.mainloop()
