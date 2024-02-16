# Decision-Tree
An application which draw a decision tree


Potrzebne biblioteki :
anytree,
pandas.

Potrzebne pobrane i zainstalowane aplikacje:
graphviz.

program pobiera dwa pliki csv:
EsiProjekt.csv
EsiProjekt_pelny.csv

Oba te pliki zawierają dane w istotnej formie:
w drugim wierszu są pytania zakończone znakiem zapytania, a pod nimi odpowiedzi.
każdy kolejny wiersz jest uzupełniony 0 i 1, przy czym 1 oznacza prawdę, a 0 fałsz.
odpowiedzi dla wymienionej sytuacji w csv musi posiadać dokładnie jedną 1
pliki te nie mogą posiadać polskich znaków.

program ten rysuje dwa grafy zarówno w konsoli, jak i w folderze, w którym znajduje się plik.
graf.png jest grafem dla pełnych danych i jest drukowany na konsoli jako pierwszy.
graf2.png jest grafem dla danych niepełnych i jest drukowany jako drugi.

na samym końcu na konsoli pojawia się skuteczność przewidywania dla tych danych.
