
from tkinter import *
from tkinter import scrolledtext
import collections as c

def readFile(filename):
    f = open(filename,mode='r', encoding='utf-8-sig')
    textt = f.read()
    return textt

def parseString(text):
    arr = []
    tmp = ''
    for i in range(len(text)):
        char = text[i]
        if char == ' ' or char == '\t' or char == '\n' or char == '|'or char == '"'or char == '،'or char == '#':
            if len(tmp) > 0:
                arr.append(tmp)
            tmp = ''
            continue
        tmp += char
        if i == len(text) - 1:
            if len(tmp) > 0:
                arr.append(tmp)
            tmp = ''
    return arr

def generateSubStrings(splitted, numOfSplit):
    returnedList = []
    for i in range(0, len(splitted) - (numOfSplit - 1)):
        temp = ' '.join(splitted[i:i + numOfSplit])
        splitTemp=temp.split()
        valid=True
        for j in range(len(splitTemp)):
            if splitTemp[j].find('.')!=-1 or splitTemp[j].find('!')!=-1 or splitTemp[j].find('?')!=-1:
                if(j==len(splitTemp)-1):
                    temp=temp[:len(temp)-1]
                    valid=True
                else:
                    valid = False
                    break
        if valid:
            returnedList.append(temp)
    return returnedList


def getCounts(listOfString):
    return dict(c.Counter(listOfString))

def getTrigramProbabilities(trigramDictionary, bigramDictionary, probabilities):
    for trigramKey in trigramDictionary:
        splitList = trigramKey.split()
        str = splitList[0] + ' ' + splitList[1]
        probabilities[trigramKey] = trigramDictionary[trigramKey] / bigramDictionary[str]
    return probabilities

def getBigramProbabilities(unigramDictionary,bigramDictionary,probabilities):
    for bigramKey in bigramDictionary:
        splitList = bigramKey.split()
        probabilities[bigramKey] = bigramDictionary[bigramKey] / unigramDictionary[splitList[0]]
    return probabilities


text = readFile('dataTemp.txt')

splittedText = parseString(text)

unigram = generateSubStrings(splittedText, 1)
bigram = generateSubStrings(splittedText, 2)
trigram = generateSubStrings(splittedText, 3)

unigramCounts=getCounts(unigram)
bigramCounts = getCounts(bigram)
trigramCounts = getCounts(trigram)

trigramProbabilities = {}
bigramProbabilities = {}
bigramProbabilities = getBigramProbabilities(unigramCounts, bigramCounts, bigramProbabilities)
trigramProbabilities = getTrigramProbabilities(trigramCounts, bigramCounts, trigramProbabilities)


def clicked():
    st.delete('1.0', END)
    result = {}
    if(len(txt.get().split())==2):
        for trigramKey in trigramProbabilities:
            splitList = trigramKey.split()
            str = splitList[0] + ' ' + splitList[1]
            if str == txt.get():
                result[trigramKey] = trigramProbabilities[trigramKey]

    if(len(txt.get().split())==1):
        for bigramKey in bigramProbabilities:
            splitList = bigramKey.split()
            if splitList[0] == txt.get():
                result[bigramKey] = bigramProbabilities[bigramKey]

    count = 0
    for w in sorted(result, key=result.get, reverse=True):
        count += 1
        print(w, result[w])
        st.insert(INSERT, w + '\n')
        if count == 5:
            print()
            break
    return sorted(result, key=result.get, reverse=True)

window = Tk()
window.title("Search Bar ")
window.geometry('350x400')
txt = Entry(window, bd=3, width=25)
txt.focus()
txt.place(relx=0.40, rely=0.30, anchor=CENTER)

btn = Button(window, text="Search", width=10, bd=3, command=clicked)
btn.place(relx=0.77, rely=0.30, anchor=CENTER)

st = scrolledtext.ScrolledText(window, width=30, bd=3, height=7)
st.place(relx=0.55, rely=0.5, anchor=CENTER)
window.mainloop()


# def clearAndUpdate(data):
#     result_list.delete(0, END)
#     for item in data:
#         result_list.insert(END, item)
#
#
# def checkSearch(_):
#     typed = txt.get()
#     if len(typed) != 0:
#         clearAndUpdate(clicked())
#
# root = Tk()
# root.geometry("500x600")
#
# label_search = Label(root, text="Search for a word: ", )
# label_search.pack(pady=20)
#
# txt = Entry(root, width=70, )
# txt.pack()
#
# result_list = Listbox(root, width=70, height=200)
# result_list.pack(pady=40, )
#
# clearAndUpdate([])
# txt.bind("<KeyRelease>", checkSearch)
# root.mainloop()
