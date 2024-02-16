import time
import pandas as pd
from math import log2
#import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter
#from graphviz import Digraph
import os

start_time = time.time()
current_directory = os.getcwd()
#load data into a DataFrame object:
sciezka = current_directory + "\EsiProjekt_pelny.csv"
data = pd.read_csv(sciezka,sep=";")
sciezka = current_directory + "\EsiProjekt.csv"
data2 = pd.read_csv(sciezka,sep=";")
df = pd.DataFrame(data)
df2 = pd.DataFrame(data2)

def BinaryTreePrinter(tabelka, sciezka_zapisu) :  #calosc funkcji na tabie dodac
        j = True
        h = []
        tablica = []
        liczby = []
        secondlist = []
        for i in tabelka: 
                if j:
                        tabelowedrzewo = Node(i) #deklaruje zmienną która będzie rodzicem całego drzewa
                        j=False
                        h.append(tabelowedrzewo) #dodaje zmienną do listy 
                        liczby.append(0) #dodaje 0 bo ta lista jest taka, że jak ma 1 to znaczy że lewo ogarnięte, 0 to nic nie ogarnięte a 2 to i lewo i prawo ogarnięte
                        secondlist.append("X") #deklaruję X, który potem będzie zmieniony na lokalizację, gdzie jest odpowiedź na prawo
                else:
                        if i[-1] == "?": #dla pytań
                                while liczby[-1] == 2: #wywalanie wszystkich rodzicow drzewka, gdzie juz są wszystkie odpowiedzi
                                        liczby.pop()
                                        h.pop()
                                if liczby[-1] == 0: #dodawanie komentarza "tak" jeśli idzie na lewo
                                        tablica.append( Node(i, parent=h[-1], type="tak"))
                                else : #i "nie" jeśli idzie na prawo
                                        tablica.append(Node(i, parent=h[-1], type="nie"))
                                liczby.append(liczby.pop() + 1) #zwiększam w liście liczby wartość o 1, ponieważ odpowiedziano na którąś odpowiedź
                                liczby.append(0) #dodaje 0 do liczby, ponieważ jest to pytanie, więc będzie na to udzielona odpowiedź
                                h.append(tablica[-1]) #dodaję do listy kolejny fragment drzewa, który został utworzony
                                secondlist.append("X") #dodaję do secondList X, ponieważ za X zostanie później wstaeiona id adresu, który odpowiada za pójście na prawo w drzewku
                        else: # dla odpowiedzi
                                while liczby[-1] == 2: #wywalanie wszystkich rodzicow drzewka, gdzie juz są wszystkie odpowiedzi
                                        liczby.pop()
                                        h.pop()
                                if liczby[-1] == 0: #dodawanie komentarza "tak" jeśli idzie na lewo
                                        tablica.append( Node(i, parent=h[-1], type = "tak"))
                                else : #i "nie" jeśli idzie na prawo
                                        tablica.append( Node(i, parent=h[-1], type = "nie"))
                                liczby.append(liczby.pop() + 1) #zanotowanie że na pytanie została udzielona odpowiedź
                                secondlist.append("odp") #zaznaczenie w liście secondList, że w tym miejscu jest odpowiedź finalna
                                for ik in range(len(secondlist),0,-1): #szukanie ostatniego X w secondList i podmienianie go na wartość odpowiedzi
                                     if(secondlist[ik-1] == "X"):
                                          secondlist[ik-1] = len(secondlist)
                                          break
        for pre, fill, node in RenderTree(tabelowedrzewo): #drukowanie drzewa do konsoli
                print("%s%s" % (pre, node.name))
        edgeattrfunc=lambda parent, child:'style=bold, label="{}"'.format(child.type) # a to do pliku
        UniqueDotExporter(tabelowedrzewo, edgeattrfunc=edgeattrfunc).to_picture(sciezka_zapisu)
        return secondlist #zwracana jest lista, gdzie w miejscu pytnia znajduje się numer listy w której znajduje się odpowiedź jak idzie się na prawo, oraz odp tam gdzie jest odpowiedź

def DecisionPredictor(tablica,odpowiedzi,questions, secondlist): # zwracacz odpowiedzi co wychodzi dla konkretnego zestawu pytań (odpowiedzi)
    x=0 #zmienna oznaczająca gdzie jesteśmy w liście "tablica"(to ta pełna)
    while (tablica[x])[-1]=="?": #pętla do czasu dopóki wychodzą pytania
        index_of_question=questions.index(tablica[x]) #sprawdzanie idndeksu zadanego pytania.
        if odpowiedzi[index_of_question]==1: #jeśli odpowiedż na nie brzmi tak, to idź na lewo dodając 1
            x=x+1
        else: #a jak nie to idź na prawo podmieniając x na wartośc z listy secondlist
            x = secondlist[x]
    return tablica[x]
         
#do liczenia entropii warunkowej pytanie + decyzja (E)
#question_idx -> indeks pytania
#start_decision_idx -> startowy indeks decyzji (indeks wiersza Horror)
#end_decision_idx -> koncowy indeks decyzji (indeks wiersz komedio-dramat)
#chcek_true -> do potwierdzania zaprzeczania warunku
def GetConditionEntropy(data_frame,question_idx,start_decision_idx,end_decision_idx,check_true):
    length = len(data_frame.iloc[question_idx])
    set1 = list(data_frame.iloc[question_idx])
    list_of_occur=[]
    for i in range(start_decision_idx,end_decision_idx):
        decisionset = list(data_frame.iloc[i])
        list_of_occur.append(CheckHowManyRep(set1,decisionset,check_true))
    if(sum(list_of_occur)!=0):
        entropies=list(map(SimpleEntropy,map(lambda x: x/sum(list_of_occur),list_of_occur)))
        sum_of_entropies=sum(entropies)
    else:
        sum_of_entropies=0
    return [sum(list_of_occur),sum_of_entropies]

#proste liczenie entropii 
def SimpleEntropy(value):
    if value<=0:
        return value*0
    return -value*log2(value)

#funkcja pomocnicza patrzy ile jest cześci wspólnych przy potwierdzonym warunku pytania i zaprzeczenie od tego jest zmienna confirmcondition
def CheckHowManyRep(table1,table2,confirmcondition):
    mysum = 0
    if confirmcondition==True:
        for i in range(0,len(table1)):
            if table1[i]==table2[i] and table1[i]==1:
                mysum=mysum+1
    else:
        for i in range(0,len(table1)):
            if table1[i]!=table2[i] and table1[i]==0:
                mysum=mysum+1
    return mysum

#Do liczenia I entropii 
def GetEntropy(data_frame,start_idx,end_idx):
    list_of_occur=[]
    for i in range(start_idx,end_idx):
        decisionset = list(data_frame.iloc[i])
        list_of_occur.append(sum(decisionset))
    entropies=list(map(SimpleEntropy,map(lambda x: x/sum(list_of_occur),list_of_occur)))
    return sum(entropies)

def GetTheHighestEntropy(table,end_of_questions,start_idx,end_idx):
    I=GetEntropy(table,start_idx,end_idx)
    list_of_entropies = []
    for i in range(0,end_of_questions):
        rowsum=sum(table.iloc[i])
        if(rowsum==len(table.iloc[i])or rowsum==0):
           list_of_entropies.append(-1)
           continue
        entropy_true_condition = GetConditionEntropy(table,i,start_idx,end_idx,True)
        entropy_false_condition = GetConditionEntropy(table,i,start_idx,end_idx,False)
        sum_of_entropies=entropy_true_condition[0]+entropy_false_condition[0]
        E=(entropy_true_condition[0]/sum_of_entropies)*entropy_true_condition[1]+(entropy_false_condition[0]/sum_of_entropies)*entropy_false_condition[1]
        list_of_entropies.append(I-E) 
    return list_of_entropies.index(max(list_of_entropies))

def DivideTable(table,index_of_row):
    list_of_table = []
    condition=table.columns[table.iloc[index_of_row]==1]
    true_table=table.iloc[:, table.columns.isin(condition)]
    list_of_table.append(true_table)
    condition=table.columns[table.iloc[index_of_row]<1]
    false_table=table.iloc[:,table.columns.isin(condition)]
    list_of_table.append(false_table)
    return list_of_table

def CheckIfAnyDecisionIsMade(data_frame,start_idx,end_idx):
    for i in range(start_idx,end_idx):
        if(sum(data_frame.iloc[i])==len(data_frame.iloc[i])):
            return i
    return False

class Branch:
    def __init__(self, data_frame):
        self.data_frame = data_frame
        self.is_left_branch_appear = False
        self.is_right_branch_appear = False

    def set_branch_appear(self, is_left_branch_appear, is_right_branch_appear):
        self.is_left_branch_appear = is_left_branch_appear
        self.is_right_branch_appear = is_right_branch_appear

def DecisionTree(data_frame,end_of_questions,start_idx,end_idx,questions):
    list_to_create_decision_tree = []
    list_of_nodes = []
    first_node = Branch(data_frame)
    list_of_nodes.append(first_node)
    while(len(list_of_nodes)>0):
        current_node = list_of_nodes[-1]
        if(current_node.is_right_branch_appear==True and current_node.is_left_branch_appear==True):
            list_of_nodes.pop()
            continue
        if(CheckIfAnyDecisionIsMade(current_node.data_frame,start_idx,end_idx)!=False):
            decision_idx = CheckIfAnyDecisionIsMade(current_node.data_frame,start_idx,end_idx)
            list_to_create_decision_tree.append(questions[decision_idx])
            list_of_nodes.pop()
        elif(current_node.is_left_branch_appear==False):
            current_node.is_left_branch_appear=True
            index_of_question=GetTheHighestEntropy(current_node.data_frame,end_of_questions,start_idx,end_idx)
            list_to_create_decision_tree.append(questions[index_of_question])
            new_node=Branch(DivideTable(current_node.data_frame,index_of_question)[0])
            list_of_nodes.append(new_node)
        elif(current_node.is_right_branch_appear==False):
            current_node.is_right_branch_appear=True
            index_of_question=GetTheHighestEntropy(current_node.data_frame,end_of_questions,start_idx,end_idx)
            new_node=Branch(DivideTable(current_node.data_frame,index_of_question)[1])
            list_of_nodes.append(new_node)
        else:
            return list_to_create_decision_tree
    return list_to_create_decision_tree

def GetProperties(vector):
    endOfQuestions=0
    for i in vector:
        if(i[-1]!="?"):
           endOfQuestions=list(vector).index(i)
           break
    endOfDecision=len(vector)
    return [endOfQuestions,endOfDecision]

def compare_dataframes(df1, df2):
    return list(set(df1.columns) ^ set(df2.columns))

def get_unique_columns(df1, df2):
    unique_columns = compare_dataframes(df1, df2)
    unique_values = []
    for col in unique_columns:
         if col in df1.columns:
            unique_values.append(df1[col].tolist())
         else:
            unique_values.append(df2[col].tolist())
    return unique_values

new_df=df.iloc[:,2:]
myquestions=list(df.iloc[:,1])
new_df2=df2.iloc[:,2:]

list_of_end_index_quest_dec=GetProperties(myquestions)
decision_tree = (DecisionTree(new_df,list_of_end_index_quest_dec[0],list_of_end_index_quest_dec[0],list_of_end_index_quest_dec[1],myquestions))
decision_tree2 = DecisionTree(new_df2,list_of_end_index_quest_dec[0],list_of_end_index_quest_dec[0],list_of_end_index_quest_dec[1],myquestions)
sciezka = current_directory + "\graf.png"
secondlist = BinaryTreePrinter(decision_tree,sciezka)
sciezka = current_directory + "\graf2.png"
secondlist2 = BinaryTreePrinter(decision_tree2,sciezka)

testowane = get_unique_columns(new_df,new_df2)
a = 0
b = 0
for i in range(0,len(testowane)):
     if DecisionPredictor(decision_tree2,testowane[i],myquestions,secondlist2) == DecisionPredictor(decision_tree,testowane[i],myquestions,secondlist):
          a = a+1
          b = b+1
     else:
          b = b + 1

print("Skutecznosc przewidywania dla tego zbioru danych wynosi:")
print(a/b)
end_time = time.time()
print(f"Czas pracy aplikacji: {end_time - start_time} sekund.")
