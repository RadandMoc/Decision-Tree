import time
import os
import sys
import pandas as pd
from math import log2
from anytree import Node, RenderTree
from anytree.exporter import UniqueDotExporter



def binary_tree_printer(tree_list, save_path):
    """
    Drukuje drzewo binarne i zapisuje je do pliku graficznego.
    Zwraca listę wskaźników, gdzie przy pytaniu znajduje się indeks odpowiedzi przy gałęzi prawej,
    a przy decyzji – napis 'odp'.
    """
    is_first = True
    parent_stack = []
    node_list = []
    counts = []
    pointer_list = []
    
    for element in tree_list:
        if is_first:
            tree_root = Node(element)  # Ustalamy korzeń drzewa
            is_first = False
            parent_stack.append(tree_root)
            counts.append(0)  # Licznik: 0 = brak odpowiedzi, 1 = lewa, 2 = obie odpowiedzi
            pointer_list.append("X")  # 'X' zostanie później podmienione na indeks odpowiedzi prawej
        else:
            if element[-1] == "?":  # Jeżeli element to pytanie
                while counts[-1] == 2:
                    counts.pop()
                    parent_stack.pop()
                if counts[-1] == 0:  # Odpowiedź "tak" (lewa gałąź)
                    node_list.append(Node(element, parent=parent_stack[-1], type="tak"))
                else:  # Odpowiedź "nie" (prawa gałąź)
                    node_list.append(Node(element, parent=parent_stack[-1], type="nie"))
                counts.append(counts.pop() + 1)
                counts.append(0)
                parent_stack.append(node_list[-1])
                pointer_list.append("X")
            else:  # Element to odpowiedź (decyzja)
                while counts[-1] == 2:
                    counts.pop()
                    parent_stack.pop()
                if counts[-1] == 0:
                    node_list.append(Node(element, parent=parent_stack[-1], type="tak"))
                else:
                    node_list.append(Node(element, parent=parent_stack[-1], type="nie"))
                counts.append(counts.pop() + 1)
                pointer_list.append("odp")
                # Znajdź ostatni "X" i podmień na aktualny indeks (dla gałęzi prawej)
                for idx in range(len(pointer_list), 0, -1):
                    if pointer_list[idx - 1] == "X":
                        pointer_list[idx - 1] = len(pointer_list)
                        break

    # Drukowanie drzewa do konsoli
    for pre, fill, node in RenderTree(tree_root):
        print(f"{pre}{node.name}")
    
    edge_attr_func = lambda parent, child: 'style=bold, label="{}"'.format(child.type)
    UniqueDotExporter(tree_root, edgeattrfunc=edge_attr_func).to_picture(save_path)
    
    return pointer_list


def decision_predictor(tree_list, answers, questions, pointer_list):
    """
    Dla danego zestawu odpowiedzi (lista answers) zwraca decyzję
    na podstawie drzewa decyzyjnego (tree_list) oraz listy pointer_list.
    """
    index = 0
    while tree_list[index][-1] == "?":
        question_index = questions.index(tree_list[index])
        if answers[question_index] == 1:
            index += 1
        else:
            index = pointer_list[index]
    return tree_list[index]


def simple_entropy(probability):
    """
    Oblicza prostą entropię dla danej wartości prawdopodobieństwa.
    """
    if probability <= 0:
        return 0
    return -probability * log2(probability)


def check_how_many_rep(list1, list2, confirm_condition):
    """
    Zlicza wystąpienia wspólnych elementów w dwóch listach przy potwierdzonym lub zaprzeczonym warunku.
    """
    count = 0
    if confirm_condition:
        for i in range(len(list1)):
            if list1[i] == list2[i] and list1[i] == 1:
                count += 1
    else:
        for i in range(len(list1)):
            if list1[i] != list2[i] and list1[i] == 0:
                count += 1
    return count


def get_condition_entropy(df, question_idx, start_decision_idx, end_decision_idx, check_true):
    """
    Oblicza entropię warunkową dla danego pytania (question_idx) oraz zakresu decyzji.
    """
    set1 = list(df.iloc[question_idx])
    occurrences = []
    for i in range(start_decision_idx, end_decision_idx):
        decision_set = list(df.iloc[i])
        occurrences.append(check_how_many_rep(set1, decision_set, check_true))
    total = sum(occurrences)
    if total != 0:
        entropies = list(map(simple_entropy, [x / total for x in occurrences]))
        sum_of_entropies = sum(entropies)
    else:
        sum_of_entropies = 0
    return [total, sum_of_entropies]


def get_entropy(df, start_idx, end_idx):
    """
    Oblicza entropię I dla zakresu wierszy.
    """
    occurrences = []
    for i in range(start_idx, end_idx):
        decision_set = list(df.iloc[i])
        occurrences.append(sum(decision_set))
    total_occurrences = sum(occurrences)
    entropies = list(map(simple_entropy, [x / total_occurrences for x in occurrences]))
    return sum(entropies)


def get_the_highest_entropy(df, end_of_questions, start_idx, end_idx):
    """
    Zwraca indeks pytania, które ma największy przyrost entropii.
    """
    total_entropy = get_entropy(df, start_idx, end_idx)
    entropy_gains = []
    for i in range(end_of_questions):
        row_sum = sum(df.iloc[i])
        if row_sum == len(df.iloc[i]) or row_sum == 0:
            entropy_gains.append(-1)
            continue
        entropy_true = get_condition_entropy(df, i, start_idx, end_idx, True)
        entropy_false = get_condition_entropy(df, i, start_idx, end_idx, False)
        sum_occurrences = entropy_true[0] + entropy_false[0]
        conditional_entropy = (entropy_true[0] / sum_occurrences) * entropy_true[1] + (entropy_false[0] / sum_occurrences) * entropy_false[1]
        entropy_gains.append(total_entropy - conditional_entropy)
    return entropy_gains.index(max(entropy_gains))


def divide_table(df, question_row_idx):
    """
    Dzieli ramkę danych na dwie części: kolumny, gdzie w danym wierszu (pytaniu) wartość wynosi 1 (true) oraz pozostałe.
    """
    split_dfs = []
    true_columns = df.columns[df.iloc[question_row_idx] == 1]
    true_df = df.loc[:, df.columns.isin(true_columns)]
    split_dfs.append(true_df)
    false_columns = df.columns[df.iloc[question_row_idx] < 1]
    false_df = df.loc[:, df.columns.isin(false_columns)]
    split_dfs.append(false_df)
    return split_dfs


def check_if_any_decision_is_made(df, start_idx, end_idx):
    """
    Sprawdza, czy w podanym zakresie wierszy została osiągnięta pełna decyzja (wszystkie wartości równe 1).
    Zwraca indeks wiersza lub False, jeśli żadna decyzja nie została podjęta.
    """
    for i in range(start_idx, end_idx):
        if sum(df.iloc[i]) == len(df.iloc[i]):
            return i
    return False


class Branch:
    def __init__(self, df):
        self.df = df
        self.left_branch_created = False
        self.right_branch_created = False

    def set_branch_appear(self, left_branch, right_branch):
        self.left_branch_created = left_branch
        self.right_branch_created = right_branch


def decision_tree(df, end_of_questions, start_idx, end_idx, questions):
    """
    Buduje drzewo decyzyjne w postaci listy pytań/decyzji.
    """
    decision_tree_list = []
    branch_stack = []
    root_branch = Branch(df)
    branch_stack.append(root_branch)
    
    while branch_stack:
        current_branch = branch_stack[-1]
        
        if current_branch.left_branch_created and current_branch.right_branch_created:
            branch_stack.pop()
            continue
        
        decision_idx = check_if_any_decision_is_made(current_branch.df, start_idx, end_idx)
        if decision_idx is not False:
            decision_tree_list.append(questions[decision_idx])
            branch_stack.pop()
        elif not current_branch.left_branch_created:
            current_branch.left_branch_created = True
            question_idx = get_the_highest_entropy(current_branch.df, end_of_questions, start_idx, end_idx)
            decision_tree_list.append(questions[question_idx])
            new_branch = Branch(divide_table(current_branch.df, question_idx)[0])
            branch_stack.append(new_branch)
        elif not current_branch.right_branch_created:
            current_branch.right_branch_created = True
            question_idx = get_the_highest_entropy(current_branch.df, end_of_questions, start_idx, end_idx)
            new_branch = Branch(divide_table(current_branch.df, question_idx)[1])
            branch_stack.append(new_branch)
        else:
            return decision_tree_list

    return decision_tree_list


def get_properties(questions_vector):
    """
    Znajduje indeks, gdzie kończą się pytania i zaczynają decyzje.
    """
    end_of_questions = 0
    for element in questions_vector:
        if element[-1] != "?":
            end_of_questions = list(questions_vector).index(element)
            break
    end_of_decision = len(questions_vector)
    return [end_of_questions, end_of_decision]


def compare_dataframes(df1, df2):
    """
    Porównuje dwie ramki danych i zwraca listę kolumn, które występują tylko w jednej z nich.
    """
    return list(set(df1.columns) ^ set(df2.columns))


def get_unique_columns(df1, df2):
    """
    Zwraca listę unikalnych kolumn (wartości) występujących tylko w jednej z podanych ramek danych.
    """
    unique_columns = compare_dataframes(df1, df2)
    unique_values = []
    for col in unique_columns:
        if col in df1.columns:
            unique_values.append(df1[col].tolist())
        else:
            unique_values.append(df2[col].tolist())
    return unique_values


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    start_time = time.time()
    current_directory = os.getcwd()
    
    # Wczytanie danych
    full_path = os.path.join(current_directory, "EsiProjekt_pelny.csv")
    data = pd.read_csv(full_path, sep=";")
    path2 = os.path.join(current_directory, "EsiProjekt.csv")
    data2 = pd.read_csv(path2, sep=";")
    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)

    df_questions = df.iloc[:, 2:]
    questions = list(df.iloc[:, 1])
    df_questions2 = df2.iloc[:, 2:]

    question_decision_indices = get_properties(questions)
    
    # Budowa drzew decyzyjnych
    decision_tree_result = decision_tree(
        df_questions,
        question_decision_indices[0],
        question_decision_indices[0],
        question_decision_indices[1],
        questions
    )
    decision_tree_result2 = decision_tree(
        df_questions2,
        question_decision_indices[0],
        question_decision_indices[0],
        question_decision_indices[1],
        questions
    )
    
    # Generowanie grafów drzewa
    graph_path = os.path.join(current_directory, "graf.png")
    pointer_list = binary_tree_printer(decision_tree_result, graph_path)
    graph_path2 = os.path.join(current_directory, "graf2.png")
    pointer_list2 = binary_tree_printer(decision_tree_result2, graph_path2)

    # Porównanie wyników predykcji
    unique_columns = get_unique_columns(df_questions, df_questions2)
    correct = 0
    total = 0
    for test in unique_columns:
        prediction1 = decision_predictor(decision_tree_result, test, questions, pointer_list)
        prediction2 = decision_predictor(decision_tree_result2, test, questions, pointer_list2)
        if prediction1 == prediction2:
            correct += 1
        total += 1

    print("Skuteczność przewidywania dla tego zbioru danych wynosi:")
    print(correct / total)
    end_time = time.time()
    print(f"Czas pracy aplikacji: {end_time - start_time} sekund.")
