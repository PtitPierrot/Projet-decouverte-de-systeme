import numpy as np
import matplotlib.pyplot as plt

"""
On va commencer par définir des petites méthodes utiles:
_ extraire(f) sert simplement a charger un fichier .txt
_ voir(f) permet de visualiser un tableau 
_ moyenne_spe(M) :
    - M est un tableau (array)
    outcomes :
        - la moyenne du tableau sans prendre en compte les 0 
Puisque dans notre cas les 0 exacts proviennent du calcul avec
la fenêtre glissante et ne doivent donc pas être pris en compte 
dans le calcul de la moyenne.
        
"""
def extraire(f):
    T=np.loadtxt(f)
    return(T)
def moyenne_spe(M):
    m=0
    c=0
    taille=np.shape(M)
    for i in range(taille[0]):
        for j in range(taille[1]):
            if M[i][j]!=0:
                m+=M[i][j]
                c+=1
    return(m/c)
def voir(f):
    plt.imshow(P)
    plt.contour(P, colors='Red')

SC = extraire('sin_card.txt')

P  = extraire('plan.txt')

"""
Le epsilon suivant est à adapter à chaque situation. On
s'en servira par la suite pour dire qu'une valeur val tq
-epsilon < val < epsilon vaut 0.
"""
epsilon=0.0000000000000001           #a adapter a chaque situation


"""
Le BPI caractérise la position d’une profondeur par rapport à son voisinage. 
Il s’agit de calculer la différence entre une profondeur et la profondeur moyenne
de son voisinage. Lorsque l’index de position bathymétrique est positif,
le point est situé au dessus de ses voisins, et inversement. On s'intéressera 
à 4 voisinnage différents. Rectangluaire, disque, anneau, secteur.
"""
"""
Ici on considere un voisinnage rectangulaire.
arguments :
    img (np.array) --> liste de liste des profondeurs par balayage
    epsilon --> definit un intervalle dans lequel toutes les valeurs comprises sont prises en compte comme des 0
return :
    matrice des BPI avec 0

"""

def bpi_rectangle_naif(img,eps):
    L=[]
    weight = np.array([[1, 1, 1,1,1], [1,1, 0, 1,1], [1,1,1, 1, 1]])
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt = img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)
#print(bpi_rectangle_naif(P,epsilon))
#print(bpi_rectangle_naif(P,epsilon).shape)
#print(P.shape)


def bpi_disque_naif(img,eps):
    L=[]
    weight = np.array([[0,1, 1, 1,0], [1,1,1,1,1], [1,1, 0, 1,1], [1, 1, 1,1,1],[0,1,1,1,0]])
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)
#print(bpi_disque_naif(P,epsilon))

def bpi_anneau_naif(img,eps):
    L=[]
    weight = np.array([[0,0,1, 1, 1,0,0], [0,1,1,0,1,1,0], [1,1, 0,0,0, 1,1], [1,0,0,0,0,0,1],[1,1,0,0,0,1,1],[0,1,1,0,1,1,0],[0,0,1,1,1,0,0]])
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)
#print(bpi_anneau_naif(P,epsilon))

def bpi_direction_naif(img,eps,direction,rayon):
    L=[]
    weight = np.array([[0,0,0, 0, 0,0,0], [0,0,0,0,0,1,0], [0,0, 0,0,1, 1,1], [0,0, 0,0,1, 1,1],[0,0,0, 0, 0,0,0],[0,0,0, 0, 0,0,0],[0,0,0, 0, 0,0,0]])
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)
#print(bpi_direction_naif(P,epsilon))

def rectangle(longueur,hauteur):
    R=np.ones((hauteur,longueur))
    R[hauteur // 2][longueur // 2]=0
    return(R)

#print(rectangle(5,3))

def disque(rayon):
    R=np.ones((rayon*2+1,rayon*2+1))
    l,c=R.shape
    l_m=l//2
    c_m=c//2
    R[l_m][c_m]=0
    for i in range(l):
        for j in range(c):
            if abs(i-l_m)+abs(j-c_m)>rayon+1:
                R[i][j]=0
    return R

#print(disque(3))

def anneau(rayon_ext,rayon_int):
    R=disque(rayon_ext)
    l,c=R.shape
    l_m=l//2
    c_m=c//2
    R[l_m][c_m]=0
    for i in range(l):
        for j in range(c):
            if abs(i-l_m)+abs(j-c_m)<rayon_int+1:
                R[i][j]=0
    return R

#print(anneau(3,2))

def secteur(direction,rayon):
    direction=direction.lower()
    R=disque(rayon)
    l,c=R.shape
    l_m=l//2
    c_m=c//2
    if direction == 'ouest':
        for i in range(l):
            for j in range(c):
                if i>l_m:
                    R[i][j]=0
                elif j<c_m:
                    R[i][j]=0
                elif i+j<l_m+c_m:
                    R[i][j] = 0
    if direction == 'nord_ouest':
        for i in range(l):
            for j in range(c):
                if i>l_m:
                    R[i][j]=0
                elif j<c_m:
                    R[i][j]=0
                elif i+j>l_m+c_m:
                    R[i][j] = 0
    if direction == 'nord':
        for i in range(l):
            for j in range(c):
                if i>=l_m:
                    R[i][j]=0
                elif j>c_m:
                    R[i][j]=0
        d=1
        i=d
        j=0
        while j <= c_m+1:
            while i<= l_m+2:
                R[i][j]=0
                i+=1
            d+=1
            i=d
            j+=1
    if direction == 'nord_est':
        for i in range(l):
            for j in range(c):
                if i>l_m:
                    R[i][j]=0
                elif j>c_m:
                    R[i][j]=0
        d=1
        i=0
        j=d
        while i <= l_m+1:
            while j<= c_m+2:
                R[i][j]=0
                j+=1
            d+=1
            j=d
            i+=1
    if direction == 'est':
        for i in range(l):
            for j in range(c):
                if i<l_m:
                    R[i][j]=0
                elif j>=c_m:
                    R[i][j]=0
        d=1
        i=l_m+rayon
        j=1
        while i >= l_m+rayon-1:
            while j <= c_m+1:
                R[i][j]=0
                j+=1
            d += 1
            i -=1
            j=d
    if direction == 'sud_est':
        for i in range(l):
            for j in range(c):
                if i<=l_m:
                    R[i][j]=0
                elif j>c_m:
                    R[i][j]=0
        d=0
        i=l_m
        j=0
        while i <= l_m+rayon:
            while j <= rayon-d-1:
                R[i][j]=0
                j+=1
            d+=1
            i+=1
            j=0
    if direction == 'sud':
        for i in range(l):
            for j in range(c):
                if i<=l_m:
                    R[i][j]=0
                elif j<c_m:
                    R[i][j]=0
        i=l_m
        j=c_m+1
        d=1
        while i < l_m+rayon:
            while j < c_m+rayon+1:
                R[i][j]=0
                j+=1
            d+=1
            j=c_m+d
            i+=1
    if direction == 'sud_ouest':
        for i in range(l):
            for j in range(c):
                if i<l_m:
                    R[i][j]=0
                elif j<=c_m:
                    R[i][j]=0
        i=l_m+1
        j=c_m
        d=1
        while j < c_m+rayon:
            while i < l_m+rayon+1:
                R[i][j]=0
                i+=1
            d+=1
            i=l_m+d
            j+=1
    return R

#print(secteur('Sud_est',3))

def bpi_direction_naif2(img,eps,direction='ouest',rayon=3):
    L=[]
    weight=secteur(direction,rayon)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)

def bpi_anneau_naif2(img,eps,rayon_ext=3,rayon_int=1):
    L=[]
    weight=anneau(rayon_ext,rayon_int)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)

def bpi_disque_naif2(img,eps,rayon=2):
    L=[]
    weight=disque(rayon)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)

def bpi_rectangle_naif2(img,eps,longueur=5,hauteur=3):
    L=[]
    weight = rectangle(longueur,hauteur)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt = img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            if diff >0+eps:
                L1.append(1)
            #Lorsque le point est situé au dessus de ces voisins alors on le signal par un 1
            #sinon par un -1
            elif diff <0-eps :
                L1.append(-1)
            else :
                L1.append(0)
        L.append(L1)
    return np.array(L)


#print(bpi_rectangle(P,epsilon))
#voir(bpi_anneau(P,epsilon))
#plt.show()

def bpi_direction(img,direction='ouest',rayon=3):
    L=[]
    weight=secteur(direction,rayon)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            L1.append(diff)
    return np.array(L)

def bpi_anneau(img,rayon_ext=3,rayon_int=1):
    L=[]
    weight=anneau(rayon_ext,rayon_int)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            L1.append(diff)
        L.append(L1)
    return np.array(L)

def bpi_disque(img,rayon=2):
    L=[]
    weight=disque(rayon)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt=img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            L1.append(diff)
        L.append(L1)
    return np.array(L)

def bpi_rectangle(img,longueur=5,hauteur=3):
    L=[]
    weight = rectangle(longueur,hauteur)
    h = (weight.shape[0] - 1) // 2
    w = (weight.shape[1] - 1) // 2
    for i in range(h, img.shape[0] - h):
        L1=[]
        for j in range(w, img.shape[1] - w):
            trt = img[i - h:i + h + 1, j - w:j + w + 1] * weight
            moy = moyenne_spe(trt)
            diff = img[i][j] - moy
            L1.append(diff)
        L.append(L1)
    return np.array(L)

print(np.std(P))
print(bpi_rectangle(P))