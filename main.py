import string
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QWidget, QApplication, QTextEdit, QLabel, QPushButton, QVBoxLayout, QFileDialog, \
    QHBoxLayout, QTextBrowser
from PyQt5.QtWidgets import QAction,qApp,QMainWindow
from PyQt5 import  QtWidgets

import nltk
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
import re


from rouge import Rouge


import spacy

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import networkx as nx
import matplotlib.pyplot as plt

import gensim

import numpy as np

import evaluate



from Cumle import Cumle, Model

#özel isimleri bulmak için spacy kullandım nltk yetersiz kaldı
nlp = spacy.load('C:\\Users\\BeyzaNurCansever\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\en_core_web_sm\\en_core_web_sm-3.5.0')
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
snowball=SnowballStemmer('english')
lancaster=LancasterStemmer()
stemmer = PorterStemmer()

gloveFile = "glove.6B.50d.txt"
dim=50

benzerlik_orani =[]
cumle_skoru=[]


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.init_ui()



    def init_ui(self):

        self.yazi_alani = QTextEdit()

        self.cumle_skor_qline=QtWidgets.QLineEdit()
        self.benzerlik_orani_qline=QtWidgets.QLineEdit()


        self.temizle = QPushButton("Temizle")
        self.ac = QPushButton("Aç")
        self.kaydet = QPushButton("Kaydet")
        self.benzerlik_orani_gir=QPushButton("Benzerlik Oranı Gir")
        self.cumle_skor_gir=QPushButton("Cümle Skoru Gir")

        h_box = QHBoxLayout()

        h_box.addWidget(self.temizle)
        h_box.addWidget(self.ac)
        h_box.addWidget(self.kaydet)


        v_box = QVBoxLayout()

        v_box.addWidget(self.yazi_alani)
        v_box.addWidget(self.cumle_skor_qline)
        v_box.addWidget(self.cumle_skor_gir)
        v_box.addWidget(self.benzerlik_orani_qline)
        v_box.addWidget(self.benzerlik_orani_gir)

        v_box.addLayout(h_box)

        self.setLayout(v_box)

        self.setWindowTitle("Main")
        self.cumle_skor_gir.clicked.connect(self.click)
        self.benzerlik_orani_gir.clicked.connect(self.click)
        self.temizle.clicked.connect(self.yaziyi_temizle)
        self.ac.clicked.connect(self.dosya_ac)
        self.kaydet.clicked.connect(self.dosya_kaydet)


    def click(self):
        sender = self.sender()

        if sender.text() == "Benzerlik Oranı Gir":
            #print(self.benzerlik_orani_qline.text())

            #benzerlik oranını burada alıyorum SONRA İŞLENECEK
            benzerlik_orani.append(float(self.benzerlik_orani_qline.text()))
            self.benzerlik_orani_qline.clear()
        elif sender.text() == "Cümle Skoru Gir":
            #print(self.cumle_skor_qline.text())

            #cümle skorunu burada alıyorum SONRA İŞLENECEK
            cumle_skoru.append(float(self.cumle_skor_qline.text()))
            cumle_skor=self.cumle_skor_qline
            self.cumle_skor_qline.clear()

    def yaziyi_temizle(self):
        self.yazi_alani.clear()

    def dosya_ac(self):
        dosya_ismi = QFileDialog.getOpenFileName(self, "Dosya Aç", os.getenv("HOME"))
        metin=""
        summary=Summary()

        with open(dosya_ismi[0], "r") as file:
            #self.yazi_alani.setText(file.read())
            metin+=file.read()
            #print(type(file.read()))

           # print(file.read())

        ozet=summary.MetinTokenization(metin)
        self.yazi_alani.setText(ozet)
        #print(metin)

        #print(type(metin))
        #summary = Summary()
       # summary.printMetin(metin)

    def dosya_kaydet(self):
        dosya_ismi = QFileDialog.getSaveFileName(self, "Dosya Kaydet", os.getenv("HOME"))

        with open(dosya_ismi[0], "w") as file:
            file.write(self.yazi_alani.toPlainText())

    def yazdir(self,metin):

         print(metin)
         #self.yazi_alani.setText(metin)
         self.yazi_alani.append(str(metin))



class Menu(QMainWindow):

    def __init__(self):
        super().__init__()

        self.pencere = MainWindow()

        self.setCentralWidget(self.pencere)
        self.show()













#Algoritmaların geliştirileceği class


class Summary():
    def __init__(self):
        super().__init__()
    def MetinTokenization(self,metin):

        #metin burada geliyor puanlandırma ve nltk ile tokenization yapılacak
        wordEmbeddings = self.GloveModelYukle(gloveFile)

        Cumle_List = []
        cleaned_text=[]
        wordCount=0

        cumleler_temp=metin.split("\n")
        article=cumleler_temp[0]
        baslık_kelimeler=article.split(" ")
        sentences = sent_tokenize(metin)
        main_metin=""

        cumleler=[]



        #burada paragraftaki cümleleri parçalıyorum
        for v in cumleler_temp:

            #print(v)
            v = v.split(".")
            for line in v:

                line= line.replace("\n","")

                if(line!="" and line!=article):

                    #başlıksız paragrafın bütün cümlelerini elde ettim burada
                    cumleler.append(line)
        #burada sayi içeren kelime sayısı ve özel isim kelime sayılarını buluyorum cümlelerde

        words=[]
        for sentence in cumleler:
            main_sentence=sentence
            main_metin+="{}".format(sentence)
            for i in sentence.split(" "):
                if i!="":
                    words.append(i)

            #total kelime sayısı
            wordCount=wordCount+len(words)
            words.clear()




            #print(sentence)
            #noktalama işaretlerinden arındırılmış cümleler
            sentence=self.remove_punctuation(sentence)

            #print(sentence)

            # ozel isimden temizlenmiş cümle cumle_list
            #print(cumle_list)


            kelimeler = sentence.split(" ")

            cumle_length=len(kelimeler)

            #sayı içeren kelimelerin tespiti

            p2,newCumle=self.SayiVarMi(kelimeler,sentence)
            #print(newCumle)
            p1, cumle_list = self.OzelIsimBul(newCumle)



            #burada newcumle sayı noktalama işaretleri ve özel isimlerden arındırılmış oluyor

            p4,stem_cumle=self.PreProcessText(cumle_list,baslık_kelimeler)
            cleaned_text.append(stem_cumle)
            #print(stem_cumle)

            p1=round((p1/cumle_length),3)
            p2=round((p2/cumle_length),3)
            p4=round((p4/cumle_length),3)

            cumle = Cumle(0,0,0,0,main_sentence,p1,p2,0,p4,0)
            Cumle_List.append(cumle)



        """for x in Cumle_List:
            print(x.cumle)
            print(x.cumle_skor)
            print(x.cumle_benzerlik_orani)
            print(x.p1)
            print(x.p2)
            print(x.p4)"""

        #benzerlik matrisini elde ediyorum burada
        benzerlikMatrisi=self.BenzerlikHesapla(cleaned_text, wordEmbeddings, cumleler)

        Model_List,List_gecen_node_sayisi=self.CalculateP3(benzerlikMatrisi,benzerlik_orani[0])
        for i in range(len(Model_List)):
            Cumle_List[i].id=i
            Cumle_List[i].cumle_benzerlik_orani_gecen_node_sayisi=List_gecen_node_sayisi[i]
            Cumle_List[i].p3=Model_List[i]


        #tdf-if değerlerini bulma
        #burada txt de bulunan bütün stemmed kelimelerin tdf-if değerlerini bulmak için birleştirme işlemi yapıyorum
        list=[]
        stemmed_metin=""
        for i in cleaned_text:
            #str=" ".join(i)
            stemmed_metin = "{} {}".format(stemmed_metin, i)
        list.append(stemmed_metin)
        print(list)
        #burada tf-idf değeri en yüksek olan %10 luk kısmın sayısını alıyorum
        top_words_count=int(wordCount*0.1)



        #%10 da bulunan kelimelri alıyorum
        tfidfValues=self.tfidfValueCalculate(list)
        temp = tfidfValues[-(top_words_count):]
        top_words_list=[]
        #print(tfidfValues)
        for i in temp:
            top_words_list.append(i.word)
        p5_list=self.CalculateP5(cumleler,top_words_list)
        """for i in range(len(cumleler)):
            print(cumleler[i])
            print(p5_list[i])"""
        for i in range(len(Cumle_List)):
            Cumle_List[i].p5=p5_list[i]


        for i in Cumle_List:
            skor=(i.p1)+(i.p2)+(i.p3)+(i.p4)+(i.p5)
            i.cumle_skor=round(skor,2)
        List=[]
        sortedList=sorted(Cumle_List,key=lambda x:x.cumle_skor,reverse=True)
        ozet=""
        for i in Cumle_List:
            print(i.cumle_skor)
            print(i.cumle)
        for x in range(len(Cumle_List)):
            #print(sortList[x].cumle_skor)
            #print(sortList[x].cumle)
            if(sortedList[x].cumle_skor>=cumle_skoru[0]):
                ozet+="{}. ".format(sortedList[x].cumle)
                List.append(sortedList[x])
        #print(ozet)
        print(main_metin)
        cumle_skoru.clear()
        print(ozet)
        rouge=Rouge()
        skorlar=rouge.get_scores(ozet,main_metin)
        #print(skorlar)
        rouge_1=skorlar[0]['rouge-1']
        rouge1_r=rouge_1['r']
        rouge1_p=rouge_1['p']
        rouge1_f=rouge_1['f']
        #print(rouge1_r)
        #print(rouge1_p)
        #print(rouge1_f)
        ozet+="\n\nROUGE1-R:{}\nROUGE1-P:{}\nROUGE1-F1:{}".format(rouge1_r,rouge1_p,rouge1_f)




        self.drawGraph(benzerlikMatrisi,Cumle_List)

        return ozet


        """for x in List:
            print(x.cumle_skor)
            print(x.cumle)

            print(x.id)
            print(x.cumle_skor)
            print(x.cumle_benzerlik_orani_gecen_node_sayisi)
            print(x.p1)
            print(x.p2)
            print(x.p3)
            print(x.p4)
            print(x.p5)"""


    def drawGraph(self,benzerlikMatrisi,Cumle_List):
        G = nx.Graph()
        labelsBenzerlik = {}
        labelsSkor = {}
        for i in range(len(Cumle_List)):
            key = i
            value = Cumle_List[i].cumle_benzerlik_orani_gecen_node_sayisi

            value2 = Cumle_List[i].cumle_skor

            labelsBenzerlik[key] = value
            labelsSkor[key] = value2
            G.add_node(i, label=i + 1)

        for i in range(len(Cumle_List)):
            for j in range(i + 1, len(Cumle_List)):
                similarity = benzerlikMatrisi[i][j]
                if similarity > 0:
                    G.add_edge(i, j, weight=similarity)

        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        weights = nx.get_edge_attributes(G, 'weight')

        width = [weights[edge] for edge in G.edges()]

        nx.draw_networkx(
            G,
            pos,
            labels=labels,
            width=width,
            edge_color='gray',
            node_color='skyblue',
            node_size=500,
            font_size=10,
            font_weight='bold'
        )

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        for i in range(len(Cumle_List)):
            x, y = pos[i]
            plt.text(x, y + 0.16, s="BenzerlikNode:\n" + str(labelsBenzerlik[i]), bbox=dict(facecolor='red', alpha=0.5),
                     horizontalalignment='center')
        for i in range(len(Cumle_List)):
            x, y = pos[i]
            plt.text(x - 0.16, y, s="CümleSkoru:\n" + str(labelsSkor[i]), bbox=dict(facecolor='green', alpha=0.5),
                     horizontalalignment='center')

        plt.axis('off')
        plt.show()

    def CalculateP3(self,benzerlikMatrisi,benzerlik_orani):
        List=[]
        List_gecen_node_sayisi=[]
        for i in range(len(benzerlikMatrisi)):#burası satır
            total_baglantı_sayisi = 0
            benzerlik_oranı_gecen_node_Sayisi = 0
            for j in range(len(benzerlikMatrisi[0])):#burası sütun
                if i!=j:
                    if benzerlikMatrisi[i][j]!=0:
                        total_baglantı_sayisi+=1
                    if benzerlikMatrisi[i][j]>= benzerlik_orani:
                        benzerlik_oranı_gecen_node_Sayisi+=1
            p3=round(benzerlik_oranı_gecen_node_Sayisi/total_baglantı_sayisi,3)
            List_gecen_node_sayisi.append(benzerlik_oranı_gecen_node_Sayisi)

            List.append(p3)

        return List,List_gecen_node_sayisi


    def CalculateP5(self,cumleler,top_words_list):
        p5_list=[]
        for cumle in cumleler:
            counter=0
            for kelime in cumle.split(" "):
                for topWord in top_words_list:
                    if topWord in kelime:
                        counter+=1
            p5_list.append(round(counter/(len(cumle.split(" "))),3))
        return p5_list




    def tfidfValueCalculate(self,word_list):
        vector = TfidfVectorizer()
        tfidfVectors = vector.fit_transform(word_list)
        feature_names = vector.get_feature_names_out()

        modelList = []
        for i, document in enumerate(word_list):
            for j, feature_index in enumerate(tfidfVectors[i].indices):
                featureAD = feature_names[feature_index]
                tfidfDeger = round(tfidfVectors[i, feature_index], 3)
                model = Model(featureAD, tfidfDeger)
                modelList.append(model)





        sortedList = sorted(modelList, key=lambda model: model.value)
        return sortedList


    def remove_punctuation(self,text):
        # Tırnak işaretinden kurtuluyorum
        """text = re.sub(r'[\'"]', '', text)
        tokens = nltk.word_tokenize(text)

        # Remove punctuation tokens
        tokens = [token for token in tokens if token not in string.punctuation]

        # Reconstruct the text without punctuation
        text_without_punct = ' '.join(tokens)"""
        text = re.sub('[%s]' % re.escape("""|"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',text)
        return text

    def PreProcessText(self,cumle,baslik_kelimeler):
        baslik_kelime_count=0
        tokenize_kelimeler = nltk.word_tokenize(cumle)
        Stop_Word_Cumle_list = [kelime for kelime in tokenize_kelimeler if
                                kelime.lower() not in stop_words and kelime != "''" and kelime != ""]
        # burada kelimelerin köklerini buluyor
        kokler = self.KokBul(Stop_Word_Cumle_list)
        #print(stems)

        # cümle içinde başlıkta geçen kelime sayısının bulunması
        for x in kokler:
            for y in baslik_kelimeler:
                if x in y:
                    #print(x)
                    baslik_kelime_count += 1
        word_string = ' '.join(kokler)
        return baslik_kelime_count,word_string


    def OzelIsimBul(self,cumle):
        ozel_isim_count=0

        doc = nlp(cumle)
        for word in doc:
            if word.pos_ == 'PROPN' and len(word.text)>1:
                # özel isimleri cğmlelerden siliyorum
               # print(word.text)
                cumle = cumle.replace(word.text, "")
                ozel_isim_count += 1
        cumle=cumle.strip(' ')
        return ozel_isim_count,cumle

    def SayiVarMi(self,kelimeler,cumle):
        number_count=0
        for kelime in kelimeler:

            if (self.IsContainDigit(kelime)):
                # sayı içeren kelimleri cümleden siliyorum
                cumle = cumle.replace(kelime, "")
                number_count += 1
        cumle=cumle.strip(' ')

        return number_count,cumle

    def IsContainDigit(self, kelime):
        harfler = list(kelime)
        for harf in harfler:
            if (harf.isdigit()):
                return True
        return False

    def KokBul(self,words):
        stems=[]
        for stem in (stemmer,snowball,lancaster):
            stems=[stemmer.stem(t) for t in words]
        return stems

    def GloveModelYukle(self,gloveFile):
        wordEmbeddings = {}
        f = open(gloveFile, encoding='utf-8')
        for line in f:
            degerler = line.split()
            kelime = degerler[0]
            x = np.asarray(degerler[1:], dtype='float32')
            wordEmbeddings[kelime] = x
        f.close()
        return wordEmbeddings
    def BenzerlikHesapla(self,text,wordEmbeddings,sentences):



        vectors = []
        for i in text:
            if len(i) != 0:
                v = sum([wordEmbeddings.get(w, np.zeros((dim,))) for w in i.split()]) / (len(i.split()) + 0.001)
            else:
                v = np.zeros((dim,))
            vectors.append(v)
        benzerlik_matrisi = np.zeros([len(text), len(text)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                     benzerlik_matrisi[i][j] = \
                        cosine_similarity(vectors[i].reshape(1, dim), vectors[j].reshape(1, dim))[
                            0, 0]
        benzerlik_matrisi = np.round(benzerlik_matrisi, 3)

        print(benzerlik_matrisi)
        return benzerlik_matrisi




app = QApplication(sys.argv)

menu = Menu()


sys.exit(app.exec_())
