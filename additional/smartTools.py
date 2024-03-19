# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:03:48 2023

@author: kobe

Tools zum Sicherstellen von Performance mit Python

"""

import psutil

###############################################################################
# Speicherplatz für virtuellen Arbeitsspeicher (CPU) abrufen
###############################################################################

# Gesamten virtuellen Speicherplatz abrufen (in Bytes)
virtual_memory = psutil.virtual_memory()
total_memory_bytes = virtual_memory.total

# Aktuell belegten Speicherplatz abrufen (in Bytes)
used_memory_bytes = psutil.virtual_memory().used

# Konvertiere Bytes in Megabytes (MB) und Gigabytes (GB) für eine bessere Lesbarkeit
total_memory_mb = total_memory_bytes / (1024 * 1024)
used_memory_mb = used_memory_bytes / (1024 * 1024)
used_memory_gb = used_memory_bytes / (1024 * 1024 * 1024)

print("Gesamter virtueller Speicherplatz: {:.2f} MB".format(total_memory_mb))
print("Aktuell belegter Speicherplatz: {:.2f} MB".format(used_memory_mb))
print("Aktuell belegter Speicherplatz: {:.2f} GB".format(used_memory_gb))

###############################################################################
# Abeitsspeichernutzung durch laufende Prozesse
###############################################################################

# Informationen über alle Prozesse abrufen
all_processes = psutil.process_iter()

# Liste der Prozesse, die den RAM nutzen, erstellen
ram_using_processes = []
for process in all_processes:
    try:
        process_info = process.as_dict(attrs=['pid', 'name', 'memory_info'])
        memory_usage_mb = process_info['memory_info'].rss / (1024 * 1024)
        if memory_usage_mb > 0:
            process_info['memory_usage_mb'] = memory_usage_mb
            ram_using_processes.append(process_info)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

# Liste der RAM-nutzenden Prozesse sortieren nach Speicherverbrauch (absteigend)
ram_using_processes.sort(key=lambda x: x['memory_usage_mb'], reverse=True)

# Prozesse und deren Speicherverbrauch ausgeben
for process_info in ram_using_processes[:5]:
    print(f"PID: {process_info['pid']}, Name: {process_info['name']}, RAM Usage: {process_info['memory_usage_mb']:.2f} MB")

###############################################################################
# Instanzen werden nicht gelöscht, auch wenn man das Bezugsobjekt gleich nennt
###############################################################################

class PSO:
    instance_count = 0  # Klassenvariable, um die Anzahl der Instanzen zu zählen

    def __init__(self):
        PSO.instance_count += 1  # Bei jeder Initialisierung einer Instanz wird der Zähler erhöht

# Erstelle 5 Instanzen der Klasse PSO mit dem gleichen Namen "pso"
pso = PSO()
pso = PSO()
pso = PSO()
pso = PSO()
pso = PSO()

# Gib die Anzahl der erstellten Instanzen aus
print("Anzahl der Instanzen von PSO:", PSO.instance_count)

###############################################################################
# draw 
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

fig, ax =  plt.subplots(nrows=2, ncols=1, tight_layout=True)
# Erzeugen der x-Werte
x_values = np.linspace(0, 10, 200)

# Festlegen der Parameter für die Gaußverteilungen
mu1, sigma1 = 2, 0.25  # Erster Durchmesser von 1
mu2, sigma2 = 5, 1.4    # Zweiter Durchmesser von 5

# Erstellen der beiden Gaussglocken
glocke1 = gaussian(x_values, mu1, sigma1)
glocke1 = glocke1[:120]
glocke2 = gaussian(x_values, mu2, sigma2)

combined_curve = np.concatenate((glocke1, glocke2/1.5))
ax[0].fill_between(range(len(combined_curve)), combined_curve, color='lightgrey', alpha=0.5, hatch='///')
ax[0].plot(combined_curve, label='GAUSS - 2 peaks')

# Festlegen der Parameter für die Gaußverteilungen
mu1, sigma1 = 2, 0.1  # Erster Durchmesser von 1
mu2, sigma2 = 5, 0.3    # Zweiter Durchmesser von 5

# Erstellen der beiden Gaussglocken
glocke1 = gaussian(x_values, mu1, sigma1)
glocke1 = glocke1[:130]
glocke2 = gaussian(x_values, mu2, sigma2)
glocke2 = glocke2[70:150]

combined_curve = np.concatenate((glocke1, glocke2/1.5, glocke2/1.5))
ax[1].fill_between(range(len(combined_curve)), combined_curve, color='lightgrey', alpha=0.5, hatch='///')
ax[1].plot(combined_curve, label='DIRAC - 3 peaks')

for i in range(2):
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    ax[i].legend(loc='upper right')
    

