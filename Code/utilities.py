import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import seaborn as sns
import networkx as nx
import os
from sklearn.decomposition import PCA
from gensim.models.poincare import PoincareModel
from matplotlib.ticker import FormatStrFormatter
import random
import math
import re

def add_transitive_edges(G, max_depth=3):
    pairs = []
    for source in G.nodes():
        # BFS fino a max_depth
        for target in nx.descendants(G, source):
            try:
                d = nx.shortest_path_length(G, source, target)
            except nx.NetworkXNoPath:
                continue
            if d <= max_depth:
                pairs.append((source, target))
    return pairs

# Funzione per stampare il grafo come un albero (copiata da stackoverflow)
def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723
    Licensed under Creative Commons Attribution-Share Alike 
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    G: the graph (must be a tree)
    
    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.
    
    width: horizontal space allocated for this branch - avoids overlap with other branches
    
    vert_gap: gap between levels of hierarchy
    
    vert_loc: vertical location of root
    
    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def print_metricsNew(log, p=True):

    # Regex per estrarre i valori dalle righe nei log dei risultati prodotti dalla libreria 
    re_train = re.compile(
        r"Epoch:\s*(\d+).*?train_loss:\s*([0-9.eE+-]+)"
    )

    # Regex per righe di validation
    re_val = re.compile(
        r"Epoch:\s*(\d+).*?val_loss:\s*([0-9.eE+-]+)\s*val_roc:\s*([0-9.eE+-]+)\s*val_ap:\s*([0-9.eE+-]+)\s*val_precision:\s*([0-9.eE+-]+)\s*val_recall:\s*([0-9.eE+-]+)\s*val_f1:\s*([0-9.eE+-]+)"
    )

    # Regex valori finali test
    re_test = re.compile(
        r"Test set results:\s*test_loss:\s*([0-9.eE+-]+)\s*test_roc:\s*([0-9.eE+-]+)\s*test_ap:\s*([0-9.eE+-]+)\s*test_precision:\s*([0-9.eE+-]+)\s*test_recall:\s*([0-9.eE+-]+)\s*test_f1:\s*([0-9.eE+-]+)"
    )
    
    re_time = re.compile(
        r"Test set results:\s*time:\s*([0-9.eE+-]+)\s*test_roc:\s*([0-9.eE+-]+)\s*test_ap:\s*([0-9.eE+-]+)"
    )

    train_loss, train_roc, train_ap, train_precision, train_recall, train_f1 = [], [], [], [], [], []
    val_loss, val_roc, val_ap, val_precision, val_recall, val_f1 = [], [], [], [], [], []
    time = []
    test_loss = test_roc = test_ap = test_precision = test_recall = test_f1 = 0.0

    with open(log, "r") as f:
        for line in f:
            m = re_train.search(line)
            if m:
                train_loss.append(float(m.group(2)))

            m = re_val.search(line)
            if m:
                val_loss.append(float(m.group(2)))
                val_roc.append(float(m.group(3)))
                val_ap.append(float(m.group(4)))
                val_precision.append(float(m.group(5)))
                val_recall.append(float(m.group(6)))
                val_f1.append(float(m.group(7)))

            m = re_test.search(line)
            if m:
                test_loss = float(m.group(1))
                test_roc  = float(m.group(2))
                test_ap   = float(m.group(3))
                test_precision = float(m.group(4))
                test_recall = float(m.group(5))
                test_f1  = float(m.group(6))

            m = re_time.search(line)
            if m:
                time = float(m.group(1))

    epochs = list(range(1, len(train_loss)+1))
    if p :

        # ------------ GRAFICO TRAIN LOSS -------------------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

        # ------------ GRAFICO TRAIN ROC + AP ---------------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, val_roc,  label="val_roc")
        plt.title("Curva ROC")
        plt.xlabel("Epoch")
        plt.ylabel("ROC")
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ GRAFICO VALIDATION ROC + AP ----------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, val_f1,  label="val_f1")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ TEST RESULTS -----------------
    if p:
        print("Test metrics finali:")
        print(f"  test_loss      = {test_loss}")
        print(f"  test_roc       = {test_roc}")
        print(f"  test_ap        = {test_ap}")
        print(f"  test_precision = {test_precision}")
        print(f"  test_recall    = {test_recall}")
        print(f"  test_f1        = {test_f1}")
    return val_roc, val_ap, val_precision, val_recall, val_f1, test_roc, test_ap, test_precision, test_recall, test_f1

def print_metrics(log, p=True):

    # Regex per estrarre i valori dalle righe nei log dei risultati prodotti dalla libreria 
    re_train = re.compile(
        r"Epoch:\s*(\d+).*?train_loss:\s*([0-9.eE+-]+)"#\s*train_roc:\s*([0-9.eE+-]+)\s*train_ap:\s*([0-9.eE+-]+)"
    )

    # Regex per righe di validation
    re_val = re.compile(
        r"Epoch:\s*(\d+).*?val_loss:\s*([0-9.eE+-]+)\s*val_roc:\s*([0-9.eE+-]+)\s*val_ap:\s*([0-9.eE+-]+)"
    )

    # Regex valori finali test
    re_test = re.compile(
        r"Test set results:\s*test_loss:\s*([0-9.eE+-]+)\s*test_roc:\s*([0-9.eE+-]+)\s*test_ap:\s*([0-9.eE+-]+)"
    )
    
    re_time = re.compile(
        r"Test set results:\s*time:\s*([0-9.eE+-]+)\s*test_roc:\s*([0-9.eE+-]+)\s*test_ap:\s*([0-9.eE+-]+)"
    )

    train_loss, train_roc, train_ap = [], [], []
    val_loss, val_roc, val_ap = [], [], []
    time = []
    test_loss = test_roc = test_ap = 0.0

    with open(log, "r") as f:
        for line in f:
            m = re_train.search(line)
            if m:
                train_loss.append(float(m.group(2)))
                train_roc.append(0)#float(m.group(3)))
                train_ap.append(0)#float(m.group(4)))

            m = re_val.search(line)
            if m:
                val_loss.append(float(m.group(2)))
                val_roc.append(float(m.group(3)))
                val_ap.append(float(m.group(4)))

            m = re_test.search(line)
            if m:
                test_loss = float(m.group(1))
                test_roc  = float(m.group(2))
                test_ap   = float(m.group(3))

            m = re_time.search(line)
            if m:
                time = float(m.group(1))

    epochs = list(range(1, len(train_loss)+1))
    if p :

        # ------------ GRAFICO TRAIN LOSS -------------------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.show()

        # ------------ GRAFICO TRAIN ROC + AP ---------------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_roc, label="train_roc")
        plt.plot(epochs, val_roc,  label="val_roc")
        plt.title("Curva ROC")
        plt.xlabel("Epoch")
        plt.ylabel("ROC")
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ GRAFICO VALIDATION ROC + AP ----------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_ap, label="train_ap")
        plt.plot(epochs, val_ap,  label="val_ap")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ TEST RESULTS -----------------
    if p:
        print("Test metrics finali:")
        print(f"  test_loss = {test_loss}")
        print(f"  test_roc  = {test_roc}")
        print(f"  test_ap   = {test_ap}")
    return train_roc, train_ap, val_roc, val_ap, test_roc, test_ap

import re
from datetime import datetime

def extract_train_epoch_times(log_path):
    train_timestamps = []
    
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Epoch (\d+) \| avg train loss"
    )
    
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                timestamp_str = match.group(1)
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                train_timestamps.append(timestamp)
    
    # Calcolo differenze tra epoche consecutive
    epoch_times = []
    for i in range(len(train_timestamps) - 1):
        delta = (train_timestamps[i + 1] - train_timestamps[i]).total_seconds()
        epoch_times.append(delta)

    with open(log_path, "r") as f:
        for line in f:
            if "Optimization finished" in line:
                timestamp_str = line[:19]
                end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                last_epoch_time = (end_time - train_timestamps[-1]).total_seconds()
                epoch_times.append(last_epoch_time)
                break
    
    total_time = sum(epoch_times)
    avg_time = total_time / len(epoch_times) if epoch_times else 0
    
    return epoch_times, total_time, avg_time



def print_metricsKGE(log, p=True):

    # Regex per estrarre i valori dalle righe nei log dei risultati prodotti dalla libreria 
    re_train = re.compile(
        r"Epoch \s*(\d+).*?train loss:\s*([0-9.eE+-]+)"
    )

    # Regex per righe di validation
    re_val = re.compile(
        r"Epoch \s*(\d+).*?valid loss:\s*([0-9.eE+-]+)"
    )

    re_valMetrics = re.compile(
        r"valid MR:\s*([0-9.eE+-]+)\s*\|\s*MRR:\s*([0-9.eE+-]+)\s*\|\s*H@1:\s*([0-9.eE+-]+)"
    )

    # Regex valori finali test
    re_test = re.compile(
        r"test MR:\s*([0-9.eE+-]+)\s*\|\s*MRR:\s*([0-9.eE+-]+)\s*\|\s*H@1:\s*([0-9.eE+-]+)"
    )
    
    re_time = re.compile(
        r"test MR:\s*([0-9.eE+-]+)\s*MRR:\s*([0-9.eE+-]+)\s*H@1:\s*([0-9.eE+-]+)"
    )

    train_loss = []
    val_loss, val_MR, val_MRR, val_H1 = [], [], [], []
    time = []
    test_MR = test_MRR = test_H1 = 0.0

    with open(log, "r") as f:
        for line in f:
            m = re_train.search(line)
            if m:
                train_loss.append(float(m.group(2)))

            m = re_val.search(line)
            if m:
                val_loss.append(float(m.group(2)))

            m = re_valMetrics.search(line)
            if m:
                val_MR.append(float(m.group(1)))
                val_MRR.append(float(m.group(2)))
                val_H1.append(float(m.group(3)))

            m = re_test.search(line)
            if m:
                test_MR = float(m.group(1))
                test_MRR  = float(m.group(2))
                test_H1   = float(m.group(3))
            """
            m = re_time.search(line)
            if m:
                time = float(m.group(1))
            """
            
    epochs = list(range(1, len(train_loss)+1))
    epochs2 = list(range(1, len(val_MR)+1))
    E2 = (np.array(epochs2)*3)-1
    if p :

        # ------------ GRAFICO TRAIN LOSS -------------------
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ GRAFICO VALIDATION MR -------------------
        plt.figure(figsize=(7, 4))
        plt.plot(epochs2, val_MR)
        plt.title("Validation MR")
        plt.xticks(epochs2, E2)
        plt.xlabel("Epoch")
        plt.ylabel("MR")
        plt.grid()
        plt.show()

        # ------------ GRAFICO VALIDATION MRR, H1 ----------
        plt.figure(figsize=(7, 4))
        plt.plot(epochs2, val_MRR, label="val_MRR")
        plt.plot(epochs2, val_H1,  label="val_H@1")
        plt.title("Validation MRR and H@1")
        plt.xticks(epochs2, E2)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ TEST RESULTS -----------------
    if p:
        print("Test metrics finali:")
        print(f"  test_MR\t= {test_MR}")
        print(f"  test_MRR\t= {test_MRR}")
        print(f"  test_H@1\t= {test_H1}")
    
    return test_MR, test_MRR, test_H1

def caricaLog(root_dir, grid, stampaLog=True, stampaLoss=False, stampaEmb=True):

    all_train_roc, all_train_ap = [], []
    all_val_roc, all_val_ap = [], []
    all_test_roc, all_test_ap = [], []

    for i in range(len(grid)):
        folder_path = os.path.join(root_dir, str(i))
        embeddings_path = os.path.join(folder_path, "embeddings.npy")
        log = os.path.join(folder_path, "log.txt")

        if not os.path.exists(folder_path):
            print(f"Cartella {folder_path} non trovata, salto.")
            continue

        if not os.path.isfile(embeddings_path):
            print(f"File embeddings.npy non trovato in {folder_path}\n")
            continue
        
        with open(log, "r") as f:
            cmd = f.readline().strip()
        
        if stampaLog:
            print(cmd)
            print("\033[1m", grid[i], "\033[0m")

        out = print_metrics(log, stampaLoss)
        
        embeddings = np.load(embeddings_path)

        all_train_roc.append(out[0])
        all_train_ap.append(out[1])
        all_val_roc.append(out[2])
        all_val_ap.append(out[3])
        all_test_roc.append(out[4])
        all_test_ap.append(out[5])

        # Plot
        if stampaEmb:
            plt.figure(figsize=(8,5))
            circle = plt.Circle((0,0), 1, color='black', fill=False)
            plt.gca().add_artist(circle)
            if len(embeddings.shape) == 1:
                # Caso array 1D
                plt.plot(embeddings)
                plt.title(f"embeddings.npy nella cartella {i} (1D)")
            elif len(embeddings.shape) == 2 and embeddings.shape[1] == 2:
                plt.scatter(embeddings[:, 0], embeddings[:, 1])
                plt.title(f"embeddings.npy nella cartella {i} (2D scatter)")
            else:
                pca = PCA(n_components=2)
                embeddings_pca = pca.fit_transform(embeddings)
                
                plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
                plt.title(f"embeddings.npy nella cartella {i} (shape {embeddings.shape})")

            plt.show()
            print("-------------------------------------------------------------------------------------------------------------------")
    return all_train_roc, all_train_ap, all_val_roc, all_val_ap, all_test_roc, all_test_ap

def caricaLogNew(root_dir, grid, stampaLog=True, stampaLoss=False, stampaEmb=True):

    all_val_roc, all_val_ap, all_val_precision, all_val_recall, all_val_f1 = [], [], [], [], []
    all_test_roc, all_test_ap, all_test_precision, all_test_recall , all_test_f1 = [], [], [], [], []

    for i in range(len(grid)):
        folder_path = os.path.join(root_dir, str(i))
        embeddings_path = os.path.join(folder_path, "embeddings.npy")
        log = os.path.join(folder_path, "log.txt")

        if not os.path.exists(folder_path):
            print(f"Cartella {folder_path} non trovata, salto.")
            continue

        if not os.path.isfile(embeddings_path):
            print(f"File embeddings.npy non trovato in {folder_path}\n")
            continue
        
        with open(log, "r") as f:
            cmd = f.readline().strip()
        
        if stampaLog:
            print(cmd)
            print("\033[1m", grid[i], "\033[0m")

        out = print_metricsNew(log, stampaLoss)
        
        embeddings = np.load(embeddings_path)

        all_val_roc.append(out[0])
        all_val_ap.append(out[1])
        all_val_precision.append(out[2])
        all_val_recall.append(out[3])
        all_val_f1.append(out[4])
        all_test_roc.append(out[5])
        all_test_ap.append(out[6])
        all_test_precision.append(out[7])
        all_test_recall.append(out[8])
        all_test_f1.append(out[9])

        # Plot
        if stampaEmb:
            plt.figure(figsize=(8,5))
            circle = plt.Circle((0,0), 1, color='black', fill=False)
            plt.gca().add_artist(circle)
            if len(embeddings.shape) == 1:
                # Caso array 1D
                plt.plot(embeddings)
                plt.title(f"embeddings.npy nella cartella {i} (1D)")
            elif len(embeddings.shape) == 2 and embeddings.shape[1] == 2:
                plt.scatter(embeddings[:, 0], embeddings[:, 1])
                plt.title(f"embeddings.npy nella cartella {i} (2D scatter)")
            else:
                pca = PCA(n_components=2)
                embeddings_pca = pca.fit_transform(embeddings)
                
                plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1])
                plt.title(f"embeddings.npy nella cartella {i} (shape {embeddings.shape})")

            plt.show()
            print("-------------------------------------------------------------------------------------------------------------------")
    return all_val_roc, all_val_ap, all_val_precision, all_val_recall, all_val_f1, all_test_roc, all_test_ap, all_test_precision, all_test_recall , all_test_f1


def caricaLogKGE(root_dir, stampaLog=True, stampaLoss=False):

    test_MR, test_MRR, test_H1, test_H3, test_H10 = [], [], [], [], []

    folder_path = os.path.join(root_dir)
    log = os.path.join(folder_path, "train.log")

    if not os.path.exists(folder_path):
        print(f"Cartella {folder_path} non trovata, salto.")
        return
    
    with open(log, "r") as f:
        cmd = f.readline().strip()

    out = print_metricsKGE(log, stampaLoss)

    with open(os.path.join(folder_path, "test.log")) as f:
        text = f.read()

    match = re.search(r"Disease-only Evaluation.*?\n.*?MR: ([\d.]+) \| MRR: ([\d.]+) \| H@1: ([\d.]+) \| H@3: ([\d.]+) \| H@10: ([\d.]+)", text)

    if match:
        mr, mrr, h1, h3, h10 = match.groups()

    test_MR.append(float(mr))
    test_MRR.append(float(mrr))
    test_H1.append(float(h1))
    test_H3.append(float(h3))
    test_H10.append(float(h10))

    times, timesTot, timesAVG = extract_train_epoch_times(log)
    #print("Tempo tot:\t", timesTot, "sec")
    #print("Tempo medio:\t", timesAVG, "sec")

    return test_MR, test_MRR, test_H1, test_H3, test_H10, times


def print_times(log, p=True):
    
    re_time = re.compile(
        r"\s*time:\s*([0-9.eE+-]+)"
    )

    re_tot = re.compile(
        r"\s*elapsed:\s*([0-9.eE+-]+)"
    )

    time = []
    total = []

    with open(log, "r") as f:
        for line in f:
            m = re_time.search(line)
            t = re_tot.search(line)
            if m:
                time.append(float(m.group(1)))
            if t:
                total.append(float(t.group(1)))

    epochs = list(range(1, len(time)+1))
    n = len(epochs)
    labels = [""] * n
    
    labels[0] = epochs[0]*5
    labels[n // 2] = epochs[n // 2]*5
    labels[-1] = epochs[-1]*5
    
    mean = np.mean(time)
    ys = np.linspace(np.min(time)-0.005, np.max(time)+0.005, 5)
    ys = np.insert(ys, 1, mean)
    ys.sort()
    
    if p :
        # ------------ GRAFICO TIME ----------
        plt.figure(figsize=(17, 5))
        plt.plot(epochs, time, label="Time")
        plt.axhline(mean, color='green', linestyle='--', alpha=0.9)
        plt.title("Time every 5 epochs")
        plt.xlabel("Epoch grouped by 5")
        plt.ylabel("Time(sec)")
        plt.xticks(epochs, labels)
        plt.yticks(ys)
        plt.legend()
        plt.grid()
        plt.show()

        # ------------ TOTAL TIME -----------------
        print(f"Tempo totale: {np.sum(time):.2f} sec")
    return time, total

def stampa_tempi(tempi, tempi2, short):
    # ========== TEMPI MEDI ========== 
    plt.figure(figsize=(13, 5))
    plt.title("Tempi medi con i vari parametri (ogni 5 epochs)")

    x = np.arange(len(short))
    width=0.4
    bars1 = plt.bar(x+width/2, [np.mean(x) for x in tempi], width, label='HGCN', color='blue')
    bars2 = plt.bar(x-width/2, [np.mean(x) for x in tempi2], width, label='GCN', color='red')

    plt.xticks(x, short)
    plt.legend()

    i = 0
    for bar in bars1:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center', va='bottom', fontsize=12
        )
        i += 1

    i = 0
    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center', va='bottom', fontsize=12
        )
        i += 1

    plt.xlabel("Parametri")
    plt.ylabel("Tempo (sec)")
    plt.ylim(0, 0.33)
    plt.show()

    # ========== TEMPI TOTALI ========== 
    plt.figure(figsize=(13, 5))
    plt.title("Tempi medi con i vari parametri (ogni 5 epochs)")

    x = np.arange(len(short))
    width=0.4
    bars1 = plt.bar(x+width/2, [np.sum(x) for x in tempi], width, label='HGCN', color='blue')
    bars2 = plt.bar(x-width/2, [np.sum(x) for x in tempi2], width, label='GCN', color='red')

    plt.xticks(x, short)
    plt.legend()

    i = 0
    for bar in bars1:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center', va='bottom', fontsize=12
        )
        i += 1

    i = 0
    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center', va='bottom', fontsize=12
        )
        i += 1

    plt.xlabel("Parametri")
    plt.ylabel("Tempo (sec)")
    plt.ylim(0, 64)
    plt.show()
    
def carica_file(root_dir, stampa, flag=False):
    
    all_test_roc = []
    all_test_ap = []
    tempi = []
    total = []

    for i in range(len(stampa)):
        folder_path = os.path.join(root_dir, str(i))
        log = os.path.join(folder_path, "log.txt")

        if not os.path.exists(folder_path):
            print(f"Cartella {folder_path} non trovata, salto.")
            continue

        with open(log, "r") as f:
            cmd = f.readline().strip()

        if flag:
            print(cmd)
            print("\033[1m", stampa[i], "\033[0m")
            print("-------------------------------------------------------------------------------------------------------------------")
        t1, t2 = print_times(log, False)
        tempi.append(t1)
        total.append(t2)

        out = print_metrics(log, False)

        all_test_roc.append(float(out[4]))
        all_test_ap.append(float(out[5]))
        
    return all_test_roc, all_test_ap, tempi, total

def carica_fileNew(root_dir, stampa, flag=False):
    
    all_test_roc, all_test_ap, all_test_precision, all_test_recall, all_test_f1 = [], [], [], [], []
    tempi = []
    total = []

    for i in range(len(stampa)):
        folder_path = os.path.join(root_dir, str(i))
        log = os.path.join(folder_path, "log.txt")

        if not os.path.exists(folder_path):
            print(f"Cartella {folder_path} non trovata, salto.")
            continue

        with open(log, "r") as f:
            cmd = f.readline().strip()

        if flag:
            print(cmd)
            print("\033[1m", stampa[i], "\033[0m")
            print("-------------------------------------------------------------------------------------------------------------------")
        t1, t2 = print_times(log, False)
        tempi.append(t1)
        total.append(t2)

        out = print_metricsNew(log, False)

        all_test_roc.append(float(out[5]))
        all_test_ap.append(float(out[6]))
        all_test_precision.append(float(out[7]))
        all_test_recall.append(float(out[8]))
        all_test_f1.append(float(out[9]))
        
    return all_test_roc, all_test_ap, all_test_precision, all_test_recall, all_test_f1, tempi, total

def printBARS2(short, HGCN, GCN, rocHGCN, rocGCN, title):
    
    MAX = math.ceil(np.max([HGCN, GCN]) / 0.5) * 0.5
    maX = MAX*0.01
    
    x = np.arange(len(short))
    width = 0.13
    gap = 0.02

    offsets = [
        -2.5*(width+gap),
        -1.5*(width+gap),
        -0.5*(width+gap),
         0.5*(width+gap),
         1.5*(width+gap),
         2.5*(width+gap)
    ]

    plt.figure(figsize=(22, 8))
    plt.title(title)

    bars1 = plt.bar(x + offsets[0], HGCN[0], width, label='HGCN', color='blue')
    bars2 = plt.bar(x + offsets[1], HGCN[1], width, color='blue')

    bars4 = plt.bar(x + offsets[3], GCN[0], width, label='GCN', color='red')
    bars5 = plt.bar(x + offsets[4], GCN[1], width, color='red')

    plt.xticks(x, short)
    plt.ylim(0, MAX)
    plt.legend()

    j = 0
    for bars in [bars1, bars2]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{rocHGCN[j][i]:.2f}",
                ha='center', va='bottom', fontsize=11, color="white"
            )
            i += 1
        j += 1

    j = 0
    for bars in [bars4, bars5]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{rocGCN[j][i]:.2f}",
                ha='center', va='bottom', fontsize=11, color="black"
            )
            i += 1
        j += 1

    plt.margins(x=0.01)
    plt.ylim(0, MAX)
    plt.show()

def scatterP2(TimeH, TimeG, rocHGCN, rocGCN):

    Th = np.median(TimeH, axis=0)
    Tg = np.median(TimeG, axis=0)
    rocH = np.median(rocHGCN, axis=0)
    rocG = np.median(rocGCN, axis=0)

    coeffH = np.round(15+np.std(TimeH, axis=0) * 15)
    coeffG = np.round(15+np.std(TimeG, axis=0) * 15)


    colors = ["white", "#FFFA6F", "#52B112FF", "black"]
    colorsH = ["#FFFFB8", "#81CFFD", "#4F46F0", "#3519A3"]
    colorsG = ["#FFFFB8", "#E6A3A2", "#F5504D", "#951A1A"]
    texts = ["0.0005", "0.01", "0.09"]

    fig, ax = plt.subplots(figsize=(16, 8))

    xs = Th
    ys = rocH
    for j in range(len(xs)):
        if j//4 == 0:
            ax.scatter(xs[j], ys[j], marker='X', color=colors[j%4], edgecolors="blue", s=9*coeffH[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 1:
            ax.scatter(xs[j], ys[j], marker='s', color=colors[j%4], edgecolors="blue", s=9*coeffH[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 2:
            ax.scatter(xs[j], ys[j], marker='o', color=colors[j%4], edgecolors="blue", s=9*coeffH[j], linewidths=2, alpha=1, zorder=1)
        #plt.scatter(xs[j], ys[j], color=colors[j%4], lw=1.3, zorder=10)
        #plt.text(xs[j]+0.48, ys[j]-0.006, texts[j//4])

    xs = Tg
    ys = rocG
    for j in range(len(xs)):
        if j//4 == 0:
            ax.scatter(xs[j], ys[j], marker='X', color=colors[j%4], edgecolors="red", s=9*coeffG[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 1:
            ax.scatter(xs[j], ys[j], marker='s', color=colors[j%4], edgecolors="red", s=9*coeffG[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 2:
            ax.scatter(xs[j], ys[j], marker='o', color=colors[j%4], edgecolors="red", s=9*coeffG[j], linewidths=2, alpha=1, zorder=1)
        #plt.scatter(xs[j], ys[j], color=colors[j%4], lw=1.3, zorder=10)
        #plt.text(xs[j]+0.48, ys[j]-0.006, texts[j//4])

    ax.scatter(40, 0, marker='s', edgecolors="#DBDBDB", color=colors[0], label='  dim 2')
    ax.scatter(40, 0, marker='s', edgecolors=colors[1], color=colors[1], label='  dim 3')
    ax.scatter(40, 0, marker='s', edgecolors=colors[2], color=colors[2], label='  dim 10')
    ax.scatter(40, 0, marker='s', edgecolors=colors[3], color=colors[3], label='  dim 25')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')

    ax.scatter(40, 0, marker='X', edgecolors="black", color="white", label='lr 0.005    ')
    ax.scatter(40, 0, marker='s', edgecolors="black", color="white", label='lr 0.01    ')
    ax.scatter(40, 0, marker='o', edgecolors="black", color="white", label='lr 0.09    ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')

    verts = [
        (-1.0, -0.5),
        ( 1.0, -0.5),
        ( 1.0,  0.5),
        (-1.0,  0.5),
        (-1.0, -0.5),
    ]

    scale = 0.5
    verts = [(x * scale, y * scale) for x, y in verts]

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    rect_path = Path(verts, codes)
    rect_marker = MarkerStyle(rect_path)

    ax.scatter(1, 1, marker=rect_marker, edgecolor="blue", facecolor="white", s=500, linewidths=2, label=" HGCN")
    ax.scatter(2, 1, marker=rect_marker, edgecolor="red",  facecolor="white", s=500, linewidths=2, label=" GCN")

    ax.scatter(40, 0, marker='s', edgecolors="#DBDBDB", color=colors[0], label=' ')
    ax.scatter(40, 0, marker='s', edgecolors=colors[1], color=colors[1], label=' ')
    ax.scatter(40, 0, marker='s', edgecolors=colors[2], color=colors[2], label=' ')
    ax.scatter(40, 0, marker='s', edgecolors=colors[3], color=colors[3], label=' ')

    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')

    ax.legend(
        fontsize=13,
        borderpad=0.8,
        labelspacing=0.8,
        markerscale=1.7, 
        ncol=2, 
        columnspacing = 0,
    )

    ax.axhline(0.5, color='red', linestyle='--', alpha=0.9)
    ax.set_title("Confronto MEDIO tra tempo e ROC (HGCN vs GCN)", fontsize=14)
    ax.set_ylabel("ROC", fontsize=13)
    ax.set_xlabel("Tempo totale (sec)", fontsize=13)
    ax.set_yscale('logit')
    yticks = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    ax.set_ylim(0.23, 0.98)
    ax.set_xlim(22, 59)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks])
    ax.minorticks_off()
    ax.grid(True, which="both", alpha=0.3)
    plt.show()

def conPareto(SUMtimeHGCN, rocHGCN, SUMtimeGCN, rocGCN):
    finale = pd.DataFrame(columns=["Model", "Time", "ROC", "Seed", "Dim", "lr"])

    lrs = [0.0005, 0.01, 0.9]
    dims = [2, 3, 10, 25]

    for j in range(3):
        for i in range(len(rocHGCN[j])):
            finale = pd.concat([pd.DataFrame([["HGCN", SUMtimeHGCN[j][i], rocHGCN[j][i], j+55, dims[i%4], lrs[i//4]]], columns=finale.columns), finale], ignore_index=True)
            finale = pd.concat([pd.DataFrame([["GCN", SUMtimeGCN[j][i], rocGCN[j][i], j+55, dims[i%4], lrs[i//4]]], columns=finale.columns), finale], ignore_index=True)

    plt.figure(figsize=(16, 8))

    def get_pareto_frontier(times, rocs):
        data = pd.DataFrame({'Time': times, 'ROC': rocs}).sort_values(by='Time')
        
        pareto_front = []
        current_max_roc = -np.inf
        
        for _, row in data.iterrows():
            if row['ROC'] > current_max_roc:
                pareto_front.append((row['Time'], row['ROC']))
                current_max_roc = row['ROC']
                
        return zip(*pareto_front)

    df_hgcn = finale[finale["Model"]=="HGCN"]
    df_gcn = finale[finale["Model"]=="GCN"]

    plt.scatter(df_gcn['Time'], df_gcn['ROC'], color='red', alpha=0.3, label='GCN (All)', s=100)
    plt.scatter(df_hgcn['Time'], df_hgcn['ROC'], color='blue', alpha=0.3, label='HGCN (All)', s=100)

    # Frontiera di pareto per GCN
    p_time_gcn, p_roc_gcn = get_pareto_frontier(df_gcn['Time'], df_gcn['ROC'])
    plt.plot(p_time_gcn, p_roc_gcn, color='red', marker='o', linestyle='-', linewidth=2, label='Pareto GCN', markersize=10)

    # Frontiera di pareto per HGCN
    p_time_hgcn, p_roc_hgcn = get_pareto_frontier(df_hgcn['Time'], df_hgcn['ROC'])
    plt.plot(p_time_hgcn, p_roc_hgcn, color='blue', marker='o', linestyle='-', linewidth=2, label='Pareto HGCN', markersize=10)

    plt.yscale('logit')
    ticks = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    plt.yticks(ticks, [str(t) for t in ticks])

    plt.title('Frontiera di Pareto: Efficienza HGCN vs GCN', fontsize=14)
    plt.xlabel('Tempo totale (sec)')
    plt.ylabel('ROC (Scala Logit)')
    plt.legend()
    plt.minorticks_off()
    plt.xlim(23, 60)
    plt.grid(True, which="both", alpha=0.3)
    plt.show()        


def scatterP(TimeH, TimeG, rocHGCN, rocGCN, l=5, title="", yname=""):

    Th = np.median(TimeH, axis=0)
    Tg = np.median(TimeG, axis=0)
    rocH = np.median(rocHGCN, axis=0)
    rocG = np.median(rocGCN, axis=0)

    minX = np.min([TimeG, TimeH])
    maxX = np.max([TimeG, TimeH])

    maxStd = np.max([np.std(rocHGCN, axis=0), np.std(rocGCN, axis=0)])

    coeffH = np.round(30 + np.std(rocHGCN, axis=0)/maxStd*100)
    coeffG = np.round(30 + np.std(rocGCN, axis=0)/maxStd*100)


    colors = ["white", "yellow", "#BF00FF", "black"]
    colorsH = ["#D8EDFF", "#2995D4E6", "#3B31F6", "#2A0F98"]
    colorsG = ["#FEC2BC", "#FF5757FF", "#C42523", "#7A0E0E"]
    texts = ["0.0005", "0.01", "0.09"]

    fig, ax = plt.subplots(figsize=(16, 8))

    xs = Th
    ys = rocH
    for j in range(len(xs)):
        if j//4 == 0:
            ax.scatter(xs[j], ys[j], marker='X', color=colorsH[j%4], edgecolors=colorsH[j%4], s=9*coeffH[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 1:
            ax.scatter(xs[j], ys[j], marker='s', color=colorsH[j%4], edgecolors=colorsH[j%4], s=9*coeffH[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 2:
            ax.scatter(xs[j], ys[j], marker='o', color=colorsH[j%4], edgecolors=colorsH[j%4], s=9*coeffH[j], linewidths=2, alpha=1, zorder=1)
        #plt.scatter(xs[j], ys[j], color=colors[j%4], lw=1.3, zorder=10)
        #plt.text(xs[j]+0.48, ys[j]-0.006, texts[j//4])

    xs = Tg
    ys = rocG
    for j in range(len(xs)):
        if j//4 == 0:
            ax.scatter(xs[j], ys[j], marker='X', color=colorsG[j%4], edgecolors=colorsG[j%4], s=9*coeffG[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 1:
            ax.scatter(xs[j], ys[j], marker='s', color=colorsG[j%4], edgecolors=colorsG[j%4], s=9*coeffG[j], linewidths=2, alpha=1, zorder=1)
        elif j//4 == 2:
            ax.scatter(xs[j], ys[j], marker='o', color=colorsG[j%4], edgecolors=colorsG[j%4], s=9*coeffG[j], linewidths=2, alpha=1, zorder=1)
        #plt.scatter(xs[j], ys[j], color=colors[j%4], lw=1.3, zorder=10)
        #plt.text(xs[j]+0.48, ys[j]-0.006, texts[j//4])

    ax.scatter(40, 0, marker='s', edgecolors=colorsH[0], color=colorsH[0], label='  dim 2')
    ax.scatter(40, 0, marker='s', edgecolors=colorsH[1], color=colorsH[1], label='  dim 4')
    ax.scatter(40, 0, marker='s', edgecolors=colorsH[2], color=colorsH[2], label='  dim 10')
    ax.scatter(40, 0, marker='s', edgecolors=colorsH[3], color=colorsH[3], label='  dim 25')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')

    if l == 0:
        ax.scatter(40, 0, marker='X', edgecolors="black", color="white", label='lr 0.01    ')
        ax.scatter(40, 0, marker='s', edgecolors="black", color="white", label='lr 0.09    ')
    elif l == 1:
        ax.scatter(40, 0, marker='X', edgecolors="black", color="white", label='lr 0.005    ')
        ax.scatter(40, 0, marker='s', edgecolors="black", color="white", label='lr 0.09    ')
    elif l == 2:
        ax.scatter(40, 0, marker='X', edgecolors="black", color="white", label='lr 0.005    ')
        ax.scatter(40, 0, marker='s', edgecolors="black", color="white", label='lr 0.01    ')
    else:
        ax.scatter(40, 0, marker='s', edgecolors="black", color="white", label='lr 0.01    ')
        ax.scatter(40, 0, marker='o', edgecolors="black", color="white", label='lr 0.09    ')
        ax.scatter(40, 0, marker='X', edgecolors="black", color="white", label='lr 0.005    ')
    

    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='o', edgecolors="blue", color="blue", s=125, label='HGCN')
    ax.scatter(40, 0, marker='o', edgecolors="red", color="red", s=125, label='GCN')

    ax.scatter(40, 0, marker='s', edgecolors=colorsG[0], color=colorsG[0], label=' ')
    ax.scatter(40, 0, marker='s', edgecolors=colorsG[1], color=colorsG[1], label=' ')
    ax.scatter(40, 0, marker='s', edgecolors=colorsG[2], color=colorsG[2], label=' ')
    ax.scatter(40, 0, marker='s', edgecolors=colorsG[3], color=colorsG[3], label=' ')

    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')
    ax.scatter(40, 0, marker='s', edgecolors="w", color='w', label=' ')

    ax.legend(
        fontsize=13,
        borderpad=0.8,
        labelspacing=0.8,
        markerscale=1.7, 
        ncol=2, 
        columnspacing = 0,
    )

    ax.axhline(0.5, color='red', linestyle='--', alpha=0.9)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(yname, fontsize=13)
    ax.set_xlabel("Tempo totale (sec)", fontsize=13)
    ax.set_yscale('logit')
    yticks = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    ax.set_ylim(0.23, 0.99)
    ax.set_xlim(minX-18, maxX+5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks])
    ax.minorticks_off()
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)
    plt.show()
    return

def scatterBOX(Th, Tg, rocH, rocG):

    colors = ["white", "yellow", "#BF00FF", "black"]
    texts = ["0.0005", "0.01", "0.09"]

    plt.figure(figsize=(16, 8))

    for i in range(len(Th[0])):
        xs = np.array(Th)[:, i]
        ys = np.array(rocH)[:, i]

        xs2 = np.array(Tg)[:, i]
        ys2 = np.array(rocG)[:, i]

        plt.plot(xs, ys, color='blue', marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.plot(xs2, ys2, color='red', marker='o', linestyle='-', linewidth=2, markersize=8)
    
    plt.bar([40], [0.0001], color='blue', label='HGCN')
    plt.bar([40], [0.0001], color='red', label='GCN')

    plt.axhline(0.5, color='red', linestyle='--', alpha=0.9)
    plt.title("Confronto tra tempo e ROC (HGCN vs GCN)")
    plt.ylabel("ROC")
    plt.xlabel("Tempo totale (sec)")
    plt.legend(
        fontsize=11,
        borderpad=0.8,
        labelspacing=0.8,
        markerscale=1.7
    )
    plt.yscale('logit')
    yticks = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    plt.ylim(0.23, 0.98)
    plt.yticks(yticks, labels=[str(y) for y in yticks])
    plt.minorticks_off()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

def gridsearch(color, dims, lrs, roc1, roc2, title, yname, flag = False):

    xlabel = dims
    dims = np.arange(1, len(dims)+1)

    MAX = math.ceil(np.max([roc1, roc2]) / 0.5) * 0.5
    _, ax = plt.subplots(1, 2, figsize=(17, 6))

    for i in range(3):
        scores1 = np.array([
            [x for x in roc1[i][:4]],
            [x for x in roc1[i][4:8]],
            [x for x in roc1[i][8:12]]
        ])

        scores2 = np.array([
            [x for x in roc2[i][:4]],
            [x for x in roc2[i][4:8]],
            [x for x in roc2[i][8:12]]
        ])

        for j, lr in enumerate(lrs):
            if i == 0:
                ax[0].plot(dims, scores1[j], marker='o', label=f"lr={lr}", color=color[j], lw=2.5)
                ax[1].plot(dims, scores2[j], marker='o', label=f"lr={lr}", color=color[j], lw=2.5)
            else:
                ax[0].plot(dims, scores1[j], marker='o', color=color[j], lw=2.5)
                ax[1].plot(dims, scores2[j], marker='o', color=color[j], lw=2.5)

    ax[0].set_xlabel("Embedding dim", fontsize=13)
    ax[1].set_xlabel("Embedding dim", fontsize=13)
    ax[0].set_xticks(dims, xlabel)
    ax[1].set_xticks(dims, xlabel)

    ax[0].set_ylabel(yname, fontsize=13)
    ax[1].set_ylabel(yname, fontsize=13)
    ax[0].legend(loc='upper left', fontsize=14)
    ax[1].legend(loc='upper left', fontsize=14)
    ax[0].set_ylim(0, MAX)
    ax[1].set_ylim(0, MAX)
    ax[0].set_title(title + " HGCN", fontsize=15)
    ax[1].set_title(title + " GCN", fontsize=15)

    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)


    yticks = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]

    if not flag:
        ax[0].set_yscale('logit')   
        ax[1].set_yscale('logit')

        ax[0].set_ylim(0.23, 0.99)
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels([str(x) for x in yticks])
        ax[0].minorticks_off()
        ax[0].tick_params(axis='y', which='minor', length=0)

        ax[1].set_ylim(0.23, 0.98)
        ax[1].set_yticks(yticks)
        ax[1].set_yticklabels([str(x) for x in yticks])
        ax[1].minorticks_off()
        ax[1].tick_params(axis='y', which='minor', length=0)

    if flag:
        ax[0].set_ylim(0.05, 1)
        ax[1].set_ylim(0.05, 1)
    plt.show()

def gridsearch2(color, color2, dims, lrs, roc, roc2, title):
    scores = np.array([
        [x for x in roc[:4]],
        [x for x in roc[4:8]],
        [x for x in roc[8:12]]
    ])

    plt.figure(figsize=(13, 6))

    for i, lr in enumerate(lrs):
        plt.plot(dims, scores[i], marker='o', label=f"lr={lr}", color=color[i], lw=2.5)
        
    scores = np.array([
        [x for x in roc2[:4]],
        [x for x in roc2[4:8]],
        [x for x in roc2[8:12]]
    ])

    for i, lr in enumerate(lrs):
        plt.plot(dims, scores[i], marker='o', color=color2[i], lw=2.5)

    plt.xlabel("Embedding dim")
    plt.xticks(dims)
    plt.ylabel("ROC")
    plt.ylim(0, 1)
    plt.legend()
    plt.title(title)
    plt.show()

def printBARSDoppie(short, HGCN, GCN, rocHGCN, rocGCN, title):
    
    MAX = math.ceil(np.max([HGCN, GCN]) / 0.5) * 0.5
    maX = MAX*0.01
    
    x = np.arange(len(short))
    width = 0.2
    gap = 0.01

    offsets = [
        -1.5*(width+gap),
        -0.5*(width+gap),
         0.5*(width+gap),
         1.5*(width+gap)
    ]

    plt.figure(figsize=(22, 8))
    plt.title(title)

    bars1 = plt.bar(x + offsets[0], HGCN[0], width, label='HGCN', color='blue')
    bars2 = plt.bar(x + offsets[1], HGCN[1], width, color='blue')

    bars4 = plt.bar(x + offsets[2], GCN[0], width, label='GCN', color='red')
    bars5 = plt.bar(x + offsets[3], GCN[1], width, color='red')

    plt.xticks(x, short)
    plt.ylim(0, MAX)
    plt.legend()

    j = 0
    for bars in [bars1, bars2]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{rocHGCN[j][i]*100:.0f}",
                ha='center', va='bottom', fontsize=11, color="white"
            )
            i += 1
        j += 1

    j = 0
    for bars in [bars4, bars5]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{rocGCN[j][i]*100:.0f}",
                ha='center', va='bottom', fontsize=11, color="black"
            )
            i += 1
        j += 1

    plt.margins(x=0.01)
    plt.ylim(0, MAX)
    plt.show()

def printBARS(short, HGCN, GCN, rocHGCN, rocGCN, title):
    
    MAX = math.ceil(np.max([HGCN, GCN]) / 0.5) * 0.5
    maX = MAX*0.01
    
    x = np.arange(len(short))
    width = 0.13
    gap = 0.02

    offsets = [
        -2.5*(width+gap),
        -1.5*(width+gap),
        -0.5*(width+gap),
         0.5*(width+gap),
         1.5*(width+gap),
         2.5*(width+gap)
    ]

    plt.figure(figsize=(22, 8))
    plt.title(title)

    bars1 = plt.bar(x + offsets[0], HGCN[0], width, label='HGCN', color='blue')
    bars2 = plt.bar(x + offsets[1], HGCN[1], width, color='blue')
    bars3 = plt.bar(x + offsets[2], HGCN[2], width, color='blue')

    bars4 = plt.bar(x + offsets[3], GCN[0], width, label='GCN', color='red')
    bars5 = plt.bar(x + offsets[4], GCN[1], width, color='red')
    bars6 = plt.bar(x + offsets[5], GCN[2], width, color='red')

    plt.xticks(x, short)
    plt.ylim(0, MAX)
    plt.legend()

    j = 0
    for bars in [bars1, bars2, bars3]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{rocHGCN[j][i]:.2f}",
                ha='center', va='bottom', fontsize=11, color="black"
            )
            i += 1
        j += 1

    j = 0
    for bars in [bars4, bars5, bars6]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{rocGCN[j][i]:.2f}",
                ha='center', va='bottom', fontsize=11, color="black"
            )
            i += 1
        j += 1

    plt.margins(x=0.01)
    plt.ylim(0, MAX)
    plt.show()

def andamentoBOX(hgcn, gcn, title, yname, flag=False, yflag=False):

    seeds = [55, 56, 57]
    lrs = [0.005, 0.1, 0.9]
    dims = [2, 4, 10, 25]
    roc = ['roc']

    data, data2 = [], []
    j = 0
    for seed in seeds:
        i = 0
        for p1 in lrs:
            for p2 in dims:
                row = {'seed': seed, 'lr': p1, 'dim': p2, "roc": hgcn[j][i]}
                row2 = {'seed': seed, 'lr': p1, 'dim': p2, "roc": gcn[j][i]}
                
                i += 1
                data.append(row)
                data2.append(row2)
        j += 1

    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data2)

    df_long = df.melt(id_vars=['seed', 'lr', 'dim'], 
                    value_vars=roc, 
                    var_name='roc', 
                    value_name='score')

    df_long2 = df2.melt(id_vars=['seed', 'lr', 'dim'], 
                    value_vars=roc, 
                    var_name='roc', 
                    value_name='score')

    df_long['param_combo'] = df_long.apply(lambda row: f"l{row['lr']}d{row['dim']}", axis=1)
    df_long2['param_combo'] = df_long2.apply(lambda row: f"l{row['lr']}d{row['dim']}", axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    xx = ["2", "3", "10", "25", "2", "3", "10", "25","2", "3", "10", "25"]
    MAX = math.ceil(np.max([df_long['score'], df_long2['score']]) / 0.5) * 0.5
    MIN = math.ceil(np.min([df_long['score'], df_long2['score']]) / 0.5) * 0.5

    # --- HGCN ---
    sns.boxplot(
        x='param_combo', y='score',
        data=df_long,
        color="blue",
        medianprops={"color": "yellow", "linewidth": 2},
        ax=axes[0],
        zorder=1
    )

    for i in range(0, 11):
        if i%4 == 0 and i!=0:
            axes[0].axvline(i-0.5, color="black")

    if flag:
        axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.3)

    axes[0].set_title('GridSearch HGCN '+ title)
    axes[0].set_xlabel('Dimensione embedding')
    axes[0].set_ylabel(yname)
    axes[0].tick_params(axis='x', rotation=45)
    if yflag:
        axes[0].set_ylim(MIN-((MAX-MIN)/6), MAX+20)
        axes[0].text(0.9, MIN-((MAX-MIN)/7), "lr 0.0005", fontsize=11.5)
        axes[0].text(5, MIN-((MAX-MIN)/7), "lr 0.01", fontsize=11.5)
        axes[0].text(9.1, MIN-((MAX-MIN)/7), "lr 0.09", fontsize=11.5)
    else:
        axes[0].set_ylim(0, MAX)
        axes[0].text(0.9, MAX*0.07, "lr 0.0005", fontsize=11.5)
        axes[0].text(5, MAX*0.07, "lr 0.01", fontsize=11.5)
        axes[0].text(9.1, MAX*0.07, "lr 0.09", fontsize=11.5)
    axes[0].bar(0, 0, color="blue", label="HGCN")
    axes[0].legend(loc='upper left')

    # --- GCN ---
    sns.boxplot(
        x='param_combo', y='score',
        data=df_long2,
        color="red",
        medianprops={"color": "yellow", "linewidth": 2},
        ax=axes[1],
        zorder=1
    )

    for i in range(0, 11):
        if i%4 == 0 and i!=0:
            axes[1].axvline(i-0.5, color="black")

    if flag:
        axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.3)

    axes[1].set_title('GridSearch GCN '+title)
    axes[1].set_xlabel('Dimensione embedding')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', rotation=45)
    if yflag:
        axes[1].set_ylim(MIN-((MAX-MIN)/6), MAX+5)
        axes[1].text(0.9, MIN-((MAX-MIN)/7), "lr 0.0005", fontsize=11.5)
        axes[1].text(5, MIN-((MAX-MIN)/7), "lr 0.01", fontsize=11.5)
        axes[1].text(9.1, MIN-((MAX-MIN)/7), "lr 0.09", fontsize=11.5)
    else:
        axes[1].set_ylim(0, MAX)
        axes[1].text(0.9, MAX*0.07, "lr 0.0005", fontsize=11.5)
        axes[1].text(5, MAX*0.07, "lr 0.01", fontsize=11.5)
        axes[1].text(9.1, MAX*0.07, "lr 0.09", fontsize=11.5)

    axes[0].set_xticklabels(xx, rotation=0)
    axes[1].set_xticklabels(xx, rotation=0)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].bar(0, 0, color="red", label="GCN")
    axes[1].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def boxes(grid, HGCN, GCN, title, yl, yflag=None):

    dims = ["dim 2", "dim 4", "dim 10", "dim 26", "dim 2", "dim 4", "dim 10", "dim 26", "dim 2", "dim 4", "dim 10", "dim 26"]
    
    z1 = []
    z2 = []

    for i in range(len(grid)):
        z1.append(np.array(HGCN)[:, i])
        z2.append(np.array(GCN)[:, i])

    MAX = math.ceil(np.max([z1, z2]) / 0.5) * 0.5
    MIN = math.ceil(np.min([z1, z2]) / 0.5) * 0.5

    x = np.arange(len(grid))
    width = 0.4

    offsets = [
        -(width/2+0.02),
        (width/2+0.02)
    ]

    plt.figure(figsize=(17, 7))
    box = plt.boxplot(z1, positions=x+offsets[0], widths=width, patch_artist=True)

    for patch in box['boxes']:
        patch.set_facecolor('blue')  # colore interno del box

    # Coloriamo la mediana
    for median in box['medians']:
        median.set(color="yellow", linewidth=2)
        

    box2 = plt.boxplot(z2, positions=x+offsets[1], widths=width, patch_artist=True)

    for patch in box2['boxes']:
        patch.set_facecolor('red')  # colore interno del box

    # Coloriamo la mediana
    for median in box2['medians']:
        median.set(color="yellow", linewidth=2)
        
    for i in x[:-1]:
        if i%4 == 0 and i!=0:
            plt.axvline(i-0.5, color="black")
        plt.axvline(i+0.5, color="black", linestyle="--", alpha=0.3)

    plt.bar(0, 0, color="blue", label="HGCN")
    plt.bar(0, 0, color="red", label="GCN")

    if yflag:
        plt.ylim(MIN-((MAX-MIN)/6), MAX+5)
        plt.text(1.1, MIN-((MAX-MIN)/7), "lr 0.0005", fontsize=11.5)
        plt.text(5.3, MIN-((MAX-MIN)/7), "lr 0.01", fontsize=11.5)
        plt.text(9.3, MIN-((MAX-MIN)/7), "lr 0.09", fontsize=11.5)
    else:
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        yticks = [0.0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
        plt.yscale('logit')
        plt.yticks(yticks)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        plt.text(1.1, MAX*0.11, "lr 0.0005", fontsize=13.5)
        plt.text(5.3, MAX*0.11, "lr 0.01", fontsize=13.5)
        plt.text(9.3, MAX*0.11, "lr 0.09", fontsize=13.5)
        plt.ylim(0.1, 0.99)
        plt.minorticks_off()

    plt.title(title, fontsize=14)
    plt.xticks(x, dims)
    plt.grid(False)
    plt.ylabel(yl, fontsize=13)
    plt.tick_params(axis='both', labelsize=13)
    plt.xlim(-0.5, len(x)-0.5)
    plt.margins(x=0.1, y=0.05)
    plt.legend(loc='upper left')
    plt.show()


def printBARSOne(short, HGCN, GCN, rocHGCN, rocGCN, title):
    
    MAX = math.ceil(np.max([HGCN, GCN]) / 0.5) * 0.5
    maX = MAX*0.01
    
    xO = np.arange(len(short))
    width = 0.3
    margin = 0.4
    x = xO + (xO // 4) * margin

    offsets = [
        -(width/2),
        (width/2)
    ]

    plt.figure(figsize=(20, 8))
    plt.title(title)

    bars1 = plt.bar(x + offsets[0], np.mean(HGCN, axis=0), width, label='HGCN', color='blue', zorder=1)
    bars2 = plt.bar(x + offsets[1], np.mean(GCN, axis=0), width, label='GCN', color='red', zorder=1)

    plt.xticks(x, short)
    plt.ylim(0, MAX)
    plt.legend()

    j = 0
    for bars in [bars1]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{np.mean(rocHGCN, axis=0)[i]:.2f}",
                ha='center', va='bottom', fontsize=11, color="white"
            )
            i += 1
        j += 1

    j = 0
    for bars in [bars2]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{np.mean(rocGCN, axis=0)[i]:.2f}",
                ha='center', va='bottom', fontsize=11, color="black"
            )
            i += 1
        j += 1

    for i, i2 in enumerate(x):
        r = len(np.array(HGCN)[:, i])
        for j in range(r):
            plt.scatter(i2 + offsets[0], np.array(HGCN)[:, i][j], color="black", zorder=10, alpha=0.8, s=8)
            plt.scatter(i2 + offsets[1], np.array(GCN)[:, i][j], color="black", zorder=10, alpha=0.8, s=8)

    plt.margins(x=0.01)
    plt.ylim(0, MAX)
    plt.show()

def printBARST(short, HGCN, GCN, rocHGCN, rocGCN, title):
    
    MAX = math.ceil(np.max([HGCN, GCN]) / 0.5) * 0.5
    maX = MAX*0.01
    
    x = np.arange(len(short))
    width = 0.25
    gap = 0.02


    plt.figure(figsize=(14, 6))
    plt.title(title)

    bars1 = plt.bar(x+width/2, HGCN[0], width, color='#1f4e79')
    bars2 = plt.bar(x+width/2, HGCN[1], width, color='#4a90c2')
    bars3 = plt.bar(x+width/2, HGCN[2], width, label='HGCN', color='blue')

    bars4 = plt.bar(x-width/2, GCN[0], width, color='orange')
    bars5 = plt.bar(x-width/2, GCN[1], width, color='yellow')
    bars6 = plt.bar(x-width/2, GCN[2], width, label='GCN', color='red')

    plt.xticks(x, short)
    plt.ylim(0, MAX)
    plt.legend()

    j = 0
    for bars in [bars1, bars2, bars3]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{np.mean(np.array(rocHGCN)[:, i]):.2f}",
                ha='center', va='bottom', fontsize=11, color="white"
            )
            i += 1
        j += 1

    j = 0
    for bars in [bars4, bars5, bars6]:
        i = 0
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                maX,
                f"{np.mean(np.array(rocGCN)[:, i]):.2f}",
                ha='center', va='bottom', fontsize=11, color="black"
            )
            i += 1
        j += 1

    plt.margins(x=0.01)
    plt.ylim(0, MAX)
    plt.show()