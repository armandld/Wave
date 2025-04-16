import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pdb
import os
import matplotlib.animation as animation

# Parameters
# TODO adapt to what you need (folder path executable input filename)
executable = 'Exercice5_2025_student'  # Name of the executable (NB: .exe extension is required on Windows)
repertoire = r"/Users/Sayu/Desktop/Wave"
os.chdir(repertoire)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configuration.in")

input_filename = 'configuration.in.example'  # Name of the input file

def lire_configuration():
    config_path = os.path.join(os.path.dirname(__file__), "configuration.in")
    configuration = {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier {config_path} n'existe pas.")
    
    with open(config_path, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne = ligne.strip()
            if ligne and "=" in ligne and not ligne.startswith("#"):
                cle, valeur = ligne.split("=", 1)
                configuration[cle.strip()] = valeur.strip()
    
    return configuration

def ecrire_configuration(nouvelles_valeurs):
    """Écrit les nouvelles valeurs dans le fichier de configuration."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Le fichier {CONFIG_FILE} n'existe pas.")

    lignes_modifiees = []
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne_strippée = ligne.strip()
            if ligne_strippée and "=" in ligne_strippée and not ligne_strippée.startswith("#"):
                cle, _ = ligne_strippée.split("=", 1)
                cle = cle.strip()
                if cle in nouvelles_valeurs:
                    ligne = f"{cle} = {nouvelles_valeurs[cle]}\n"
            lignes_modifiees.append(ligne)

    with open(CONFIG_FILE, "w", encoding="utf-8") as fichier:
        fichier.writelines(lignes_modifiees)

# Paramètres physiques actuels (issus du fichier configuration.in)
cb_gauche = "libre"
cb_droite = "fixe"
v_uniform = True
A = 0.1
f_hat = 1.0
x1 = 3.0
x2 = 8.0
tfin = 1.0
equation_type = "B"
nx = 64
n_init = 3
initialization = "mode"
initial_state = "static"
CFL = 1.0
nsteps = 40
impose_nsteps = True
output = "test.txt"
n_stride = 1
ecrire_f = 1
hL = 8000.0
hR = 20.0
h00 = 4.0
xa = 450e3
xb = 950e3
L = 15.0
om = 0.1

valeurs = lire_configuration()

def actualise_valeur():
    global cb_gauche, cb_droite, v_uniform, A, f_hat, x1, x2, tfin, equation_type
    global nx, n_init, initialization, initial_state, CFL, nsteps, impose_nsteps
    global output, n_stride, ecrire_f, hL, hR, h00, xa, xb, L, om

    cb_gauche = valeurs.get("cb_gauche")
    cb_droite = valeurs.get("cb_droite")
    v_uniform = valeurs.get("v_uniform") == "true"
    A = float(valeurs.get("A"))
    f_hat = float(valeurs.get("f_hat"))
    x1 = float(valeurs.get("x1"))
    x2 = float(valeurs.get("x2"))
    tfin = float(valeurs.get("tfin"))
    equation_type = valeurs.get("equation_type")
    nx = int(valeurs.get("nx"))
    n_init = int(valeurs.get("n_init"))
    initialization = valeurs.get("initialization")
    initial_state = valeurs.get("initial_state")
    CFL = float(valeurs.get("CFL"))
    nsteps = int(valeurs.get("nsteps"))
    impose_nsteps = valeurs.get("impose_nsteps") == "true"
    output = valeurs.get("output")
    n_stride = int(valeurs.get("n_stride"))
    ecrire_f = int(valeurs.get("ecrire_f"))
    hL = float(valeurs.get("hL"))
    hR = float(valeurs.get("hR"))
    h00 = float(valeurs.get("h00"))
    xa = float(valeurs.get("xa"))
    xb = float(valeurs.get("xb"))
    L = float(valeurs.get("L"))
    om = float(valeurs.get("om"))

def ecrire_valeur(nom, valeur):
    global valeurs
    valeurs[nom] = str(valeur)
    ecrire_configuration(valeurs)
    actualise_valeur()

def lancer_simulation(param_valeur, output_file):
    ecrire_configuration({"output": output_file})
    cmd = f"./{executable} {input_filename} output={output_file}"
    subprocess.run(cmd, shell=True)

def ecrire_valeur(nom,valeur):
    global valeurs
    valeurs[nom] = valeur
    ecrire_configuration(valeurs)
    actualise_valeur()

def lancer_simulation(theta0, output_file):
    ecrire_configuration({"theta0": theta0})
    cmd = f"./{executable} {input_filename} output={output_file}"
    subprocess.run(cmd, shell=True)

outputs = []  # Liste pour stocker les fichiers de sortie
errors = []  # Liste pour stocker les erreurs
values = []

# Charger les données
f = np.loadtxt("f_test.txt")[:, 1:] # shape: (n_time_steps, n_points)
x = np.loadtxt("x_test.txt")# shape: (n_points,)
E = np.loadtxt("en_test.txt") # shape: (n_time_steps,)

# Paramètres utiles
nt, nx = f.shape
dx = x[1] - x[0]
dt = np.loadtxt("output.out_dt") if "output.out_dt" in locals() else 0.02 # approx si pas de fichier dt
time = np.arange(nt) * dt

# Tracer E(t)
plt.figure()
plt.plot(time, E)
plt.xlabel("Temps (s)")
plt.ylabel("Énergie E(t)")
plt.title("Évolution de l'énergie")
plt.grid(True)
plt.tight_layout()


# Nouvelle figure pour l'animation
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(x[0], x[-1])
ax.set_ylim(np.min(f), np.max(f) * 1.1)
ax.set_xlabel("x (m)")
ax.set_ylabel("f(x, t)")
ax.set_title("Animation de la vague")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, f[frame, :len(x)])
    ax.set_title(f"t = {frame*dt:.2f} s")
    return line,

ani = animation.FuncAnimation(fig, update, frames=nt, init_func=init,
blit=True, interval=30)

plt.tight_layout()
plt.show()
