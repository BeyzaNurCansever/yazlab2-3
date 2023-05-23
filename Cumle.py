class Cumle:



    #p1 özel isim sayısına ait parametleri gösteriyor
    #p2 number sayısına ait parametreyi gösteriyor


    def __init__(self,id,cumle_skor,cumle_benzerlik_orani_gecen_node_sayisi,orani_gecen_node_sayisi,cumle,p1,p2,p3,p4,p5):
        self.id=id
        self.cumle=cumle
        self.cumle_benzerlik_orani_gecen_node_sayisi=cumle_benzerlik_orani_gecen_node_sayisi
        self.orani_gecen_node_sayisi=orani_gecen_node_sayisi
        self.cumle_skor=cumle_skor
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.p4=p4
        self.p5=p5



class Model:
    def __init__(self,word,value):
        self.word=word
        self.value=value