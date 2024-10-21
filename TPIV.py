#!/usr/bin/env python
# coding: utf-8

# # Chain of quarks with periodic boundary conditions

# ## The problem

# Nous allons dans ce notebook implémenter l'hamiltonien $\hat{H}$ pour une chaîne de $N_s$ quarks en une dimension :
# 
# \begin{equation}\hat{H}=\sum_{i=0}^{N-1}\hat{\vec{S}}_{i}\cdot \hat{\vec{S}}_{i+1} \label{ham}=
# \hat{\vec{S}}_{0}\cdot \hat{\vec{S}}_{1}+\hat{\vec{S}}_{1}\cdot \hat{\vec{S}}_{2}+\cdots+\hat{\vec{S}}_{N-1}\cdot \hat{\vec{S}}_{0}
# \end{equation}
# 
# avec conditions aux bords périodiques : 
# 
# $$\hat{\vec{S}}_{N}=\hat{\vec{S}}_0$$
# 
# On numérote les sites $0,1,2,...,N_s-1$ car en python, la numérotation commence à $0$. 

# $\newcommand{\ket}[1]{|#1>}$

# On sait, de part la théorie des groupes, que (je ne suis pas sûre de çela) 
# 
# $$\vec{S}_i\cdot \vec{S}_{i+1}=\sum_{j=1}^{8}\lambda_{i}^{j}\lambda_{i+1}^j$$
# 
# où $\lambda_i^j$ est la matrice $j$ de Gell-Mann agissant sur le site $i$. Un site $i$ étant représenté par un espace de Hilbert à $3$ états, i.e. une base de l'espace de Hilbert $\Bbb{H}$ à un site est $\Bbb{C}^3\propto \{\ket{0},\ket{1},\ket{2}\}$.  Les matrice de Gell-Mann $\lambda_i^j$ sont de taille $3\times 3$. 
# 
# The Gell-Mann matrices are the generators of the Lie algebra $\mathfrak{su}(3)$ of SU$(3)$, denoted and defined by $\mathfrak{su}(3)=\{M_3(\Bbb{C}):M^{\dagger}+M=0 \text{ and Tr}(M)=0)\}$
# 
# L'hamiltonien $\hat{H}$ agit sur l'espace de Hilbert total $\Bbb{H}_{tot}=\otimes_{k=1}^{N_s}\Bbb{H}$ qui est dont de dimension ${3^{N_s}}$. Chaque opérateur $\lambda_i^j$ doit être interprété comme agissant sur l'espace de Hilbert total 
# $$\lambda_i^j=\hat{1}\otimes \cdots \otimes \hat{1} \otimes \lambda_i^j \otimes \hat{1}\otimes \cdots \otimes \hat{1}$$
# 
# et la composition $$\lambda_i^j \lambda_{i+1}^j=\hat{1}\otimes \cdots \hat{1}\otimes \lambda_i^j\otimes \lambda_{i+1}^j\otimes \hat{1} \cdots \otimes \hat{1}  $$

# ## Implementation of the Hamiltonian

# L'idée du code est de calculer d'abord $\hat{\vec{{S}}}_{i}\cdot \hat{\vec{{S}}}_{i+1}$ pour tout $i$, et ensuite de calculer $\hat{H}$ à partir des résultats précédents en utilisant l'équation definissant l'hamiltonien.

# On importe le package numpy qui nous sera utile pour manipuler des arrays

# In[1]:
from datetime import datetime

start = datetime.now()

def generaltwositesoperator(i,j,A,B):
    
    '''Return the tensorial product of identity, exepect at position i where there is A and at position j
    where there is B, i<j'''
    
    if i>=j:
        raise Exception("The argument in generaltwosites operator are not given correctly")
    else:
        result=np.kron(np.eye(3**i),A)
        result=np.kron(result,np.eye(3**(j-i-1)))
        if j==number_of_sites-1:
            return np.kron(result,B)
        else:
            result=np.kron(result,B)
            result=np.kron(result,np.eye(3**(number_of_sites-j-1)))
            return result


# In[2]:


import numpy as np
import matplotlib.pylab as plt
import math 

#np.set_printoptions(threshold=np.inf) ----- pour printer les matrices sans troncature

#creation des matrices de Gell-Mann
lambda1=np.array([[0,1,0],[1,0,0],[0,0,0]])
lambda2=np.array([[0,-1j,0],[1j,0,0],[0,0,0]])
lambda3=np.array([[1,0,0],[0,-1,0],[0,0,0]])
lambda4=np.array([[0,0,1],[0,0,0],[1,0,0]])
lambda5=np.array([[0,0,-1j],[0,0,0],[1j,0,0]])
lambda6=np.array([[0,0,0],[0,0,1],[0,1,0]])
lambda7=np.array([[0,0,0],[0,0,-1j],[0,1j,0]])
lambda8=np.array([[3**(-0.5),0,0],[0,3**(-0.5),0],[0,0,-2*3**(-0.5)]])
lambda_array=[lambda1,lambda2,lambda3,lambda4,lambda5,lambda6,lambda7,lambda8]

#bases de sl(3,C) proposée par B.Hall
base1=np.array([[1,0,0],[0,-1,0],[0,0,0]])
base2=np.array([[0,0,0],[0,1,0],[0,0,-1]])
base3=np.array([[0,1,0],[0,0,0],[0,0,0]])
base4=np.array([[0,0,0],[0,0,1],[0,0,0]])
base5=np.array([[0,0,1],[0,0,0],[0,0,0]])
base6=np.array([[0,0,0],[1,0,0],[0,0,0]])
base7=np.array([[0,0,0],[0,0,0],[0,1,0]])
base8=np.array([[0,0,0],[0,0,0],[1,0,0]])
base_array=[base1,base2,base3,base4,base5,base6,base7,base8]

#for i in lambda_array:
#    print(i)

#matrice identité de taille 3*3, ss stands for single site
identity_ss=np.eye(3) 

#nombre de sites
number_of_sites=2

#matrice carré nulle de taille 3**n 
zero_matrix=np.zeros((3**number_of_sites,3**number_of_sites))

#fonction qui prend deux matrices en arguments, et un indice i, et renvoie AB agissant sur l'espace de Hilbert tot

def twositesoperator(i,A,B):
    
    '''prend en argument deux matrices A,B et un entier i, et renvoie le produit tensoriel avec que des matrices
    identités, sauf en position i où il y a A, et en position i+1 où il y a B, ne traite pas le cas S_NS_0.'''
    
    operator_result=zero_matrix  

    if i >=number_of_sites-1 or i<0 :
        print('Attention ! '+ str(i)+' cannot be use to index a site.') #les sites sont numérotés 0,1,...,N-1 ---- on traite le 
        #cas i = N-1 n'est pas traité dans cette fonction
        return zero_matrix
    
    elif i>=0 and i< number_of_sites-1 :
        operator_result=np.kron(np.eye(3**i),np.kron(A,B))
        operator_result=np.kron(operator_result,np.eye(3**(number_of_sites-2-i)))
        return operator_result
    
               
#tests
#diagonal_ss=np.array([[3,0,0],[0,3,0],[0,0,3]])
#plt.spy(twositesoperator(diagonal_ss,identity_ss,2))
#print(twositesoperator(diagonal_ss,identity_ss,0))

def onesiteoperator(i,A):
     
    '''prend en argument une matrice A et un entier i, et renvoie le produit tensoriel avec que des matrices
    identités, sauf en position i où il y a A'''
    
    operator_result=zero_matrix
   
    if i>=0 or i<=number_of_sites-1:
        operator_result=np.kron(np.kron(np.eye(3**i),A),np.eye(3**(number_of_sites-1-i)))
        return operator_result
    else:
        print('Attention ! '+ str(i)+' cannot be use to index a site.') #les sites sont numérotés 0,1,...,N-1
        return zero_matrix


# In[3]:


#onesiteoperator(0,np.array([[2,2,2],[2,2,2],[2,2,2]]))


# Nous avons donc un programme qui calcul $\lambda_i^j\lambda_{i+1}^j$ pour tout $i\in \{0,1,2,...,N_s-2\}$. Pour ces $i$, que vaut $\hat{\vec{S}}_{i}\cdot \hat{\vec{S}}_{i+1}$ ? Il faut utiliser $\vec{S}_i\cdot \vec{S}_{i+1}=\sum_{j=1}^{8}\lambda_{i}^{j}\lambda_{i+1}^j$.

# In[4]:


def sdots(i):
    '''calcul le produid S_i dot S_{i+1}, pour i = 0,1,2,...,N_s-2'''
    
    sdots_i=np.zeros((3**(number_of_sites),3**(number_of_sites)))
    for k in range(0,8):
        sdots_i=sdots_i+twositesoperator(i,lambda_array[k],lambda_array[k])
    return sdots_i


# Nous avions dit en commentaire de code que nous codons $\hat{\vec{S}}_{N_s}\cdot \hat{\vec{S}}_{0}$ à part. C'est ce que nous faisons dans la prochaine cellule. Notons que 
# $$\hat{\vec{S}}_{N_s}\cdot \hat{\vec{S}}_{0}=\sum_{j=1}^{8}\lambda_{N_s}^j\lambda_0^j$$

# In[5]:


sn_dot_s0=np.zeros((3**(number_of_sites),3**(number_of_sites)))
for k in range(0,8):
    sn_dot_s0+=np.kron(lambda_array[k],np.kron(np.eye(3**(number_of_sites-2)),lambda_array[k])).real


# In[6]:


from random import seed
from random import random
# seed random number generator
#seed(1)


# In[7]:



hamiltonien=np.zeros((3**number_of_sites,3**number_of_sites))
for l in range(0,number_of_sites-1):
    hamiltonien=hamiltonien+sdots(l)


hamiltonien=hamiltonien+sn_dot_s0

#print(hamiltonien)
#plt.spy(hamiltonien)


# ## Block Diagonalisation of the hamiltonian

# Consider the Lie algebra $\mathfrak{su}(3)$ of the matrix Lie group SU$(3)$. Since the Matrix Lie group SU$(3)$ is a simply connected matrix Lie group, finding a representation of its Lie algebra $\mathfrak{su}(3)$ is totally equivalent to finding a representation of SU$(3)$.
# We want to find a representation of $\mathfrak{su}(3)$ and to do this, our goal is to find $\rho: \mathfrak{su}(3)\rightarrow \text{M}_3(C^3)$ that is a homomorphism. One can take $S^{j}=\rho(\lambda^j)=\lambda^j$ for all $j\in \{1,2,...,8\}$ acting on $\Bbb{C}^3$ and extend it to the all hilbert space by linearity. This is the fundamental representation denoted by $\textbf{3}$.
# For the moment, the basis (canonical) in which are Hamiltonian is expressed is, for $N_s$ sites $$\{\ket{0,0},\ket{0,1},\ket{0,2},\ket{1,0},\ket{1,1},\ket{1,2},...,\ket{N_s-1,0},\ket{N_s-1,1},\ket{N_s-1,2}\}$$.

# The cartan subalgebra of $\mathfrak{su}(3)$ is of dimention $2$ (from group theory) and a basis of this subalgebra can be chosen as $\{h_1:=diag(1,-1,0),h_2:=diag(0,1,-1)\}$. The hamiltonian commutes (explain this) with $h_{1_{tot}}$ and $h_{2_{tot}}$
# $$[\hat{H},h_{1_{tot}}]=[\hat{H},h_{2_{tot}}]=[h_{1_{tot}},h_{2_{tot}}]$$
# $$h_{1_{tot}}=\sum_{i=0}^{N-1}h_1^i$$ where $h_1^i$ acts on site $i$ and $$h_{2_{tot}}=\sum_{i=0}^{N-1}h_2^i$$ where $h_2^i$ acts on site $i$. 
# 

# In[8]:


#définition des opérateurs h1 et h2
h1=np.array([[1,0,0],[0,-1,0],[0,0,0]])
h2=np.array([[0,0,0],[0,1,0],[0,0,-1]])

#initialisation des operateurs h1tot et h2tot 
h1tot=np.zeros((3**number_of_sites,3**number_of_sites))
h2tot=np.zeros((3**number_of_sites,3**number_of_sites))

#calcul de h1tot et h2tot
for i in range(0,number_of_sites): #effectue number_of_sites operations, commence à 0 et finit à number_of_sites-1
    h1tot=h1tot+onesiteoperator(i,h1)
    h2tot=h2tot+onesiteoperator(i,h2)
    
#affichage de h1tot et de h2tot
#print("h1tot=")
#print(h1tot)
#print("h2tot=")
#print(h2tot)

#affichage du commutateur entre l'hamiltonien et h1tot 
#print(np.dot(hamiltonien,h1tot)-np.dot(h1tot,hamiltonien))

#affichage du commutateur entre l'hamiltonien et h2tot 
#print(np.dot(hamiltonien,h2tot)-np.dot(h2tot,hamiltonien))



# 

# Now since the two operators $\hat{H}$ and $h_{1_{tot}}$ commute, it is possible to diagonalize them simultaniously in the same basis. To do this, the first step is to find a basis for each eigenspace of one of the two operators. We choose $h_{1_{tot}}$ since it is already diagonal in the canonical basis. 
# 
# Here is one way to compute the spectrum and the associated eigenspaces of an arbitrary operator, when one has knowledge of its matrix $A$ in a basis $\{v_0,...,v_{N_s-1}\}$, and when $A$ is diagonal :
# 
# - To compute the spectrum : it is really easy since the matrix $A$ is diagonal, so the spectrum is made of all elements in the diagonal, without counting several times the same value (for example if the eigenvalue 1 appears 3 times, the spectrum only contain one time the value 1, since by definition the spectrum of an operator do not contain several times the same value)
# 
# - Compute the associated eigenspaces : regroup all the eigenvector(s) associated to the same eigenvalue, and repeat the process for all eigenvalues. Again, here, if $\lambda$ appears in the diagonal at line/column $i$, it implies that the vector $v_i$ is an eigenvector associated to the eigenvalue $\lambda$
# 
# So in the next cell, we write a programm that will do the two previous steps, given a diagonal matrix $A$ as entry.
# 

# In[9]:


def spectrum_and_eigenspace_of_diagonal_matrix(A):
    
    
    '''renvoie le spectre de A sous forme d'une liste et pour chaque valeur propre, une liste contenant les 
    numéros i des vecteurs v_i correspondants à cette valeur propre'''
    
    spectrum=[]
    eigenspaces=[[]]
    compteur=0
    basis=np.empty((1,1),dtype=int) 
    #print("calcul du spectre et des espaces propres")    
    #print("nous commencons l'itération")
        
    for i in range(0,len(A[0])):
        #print("colonne numéro ="+str(i)+ " possède élément ="+ str(A[i,i]))   
        
        if A[i,i] not in spectrum:
            
            if i==0:
            
                #print(str(A[i,i])+" n'est pas encore dans le spectre, donc on l'ajoute et le spectre est : ") 
                spectrum.append(A[i,i])
                #print(spectrum)
                for j in range(i,len(A[0])):
                    if A[j,j]==A[i,i]:
                        #print("le vecteur numéro ="+ str(j)+" est vecteur prope pour cette valeur propre :")
                        eigenspaces[i].append(j)
                        basis=np.append(basis,[[j]],axis=1)
                        #print("on l'ajoute donc à la liste des états propres associés à "+str(A[i,i]))
                        #print(eigenspaces[i])
            
            if i>0:
                
                eigenspaces.append([])
                compteur+=1
                #print(str(A[i,i])+" n'est pas encore dans le spectre, donc on l'ajoute et le spectre est : ") 
                spectrum.append(A[i,i])
                #print(spectrum)
                for j in range(i,len(A[0])):
                    if A[j,j]==A[i,i]:
                        #print("le vecteur numéro ="+ str(j)+" est vecteur prope pour cette valeur propre :")
                        eigenspaces[compteur].append(j)
                        basis=np.append(basis,[[j]],axis=1)
                        #print("on l'ajoute donc à la liste des états propres associés à "+str(A[i,i]))
                        #print(eigenspaces)
                
    result=np.delete(basis,0,axis=1) #delete element column 0 of each line
    
    return spectrum,eigenspaces,result
    
        
    
        
    


# In[10]:


#print("spectrum of h1tot is :")
#print(spectrum_and_eigenspace_of_diagonal_matrix(h1tot)[0])
#print(" ")
#print("list of eigenstates of h1tot is :")
#print(spectrum_and_eigenspace_of_diagonal_matrix(h1tot)[1])

basis_from_h1tot=[]
for i in spectrum_and_eigenspace_of_diagonal_matrix(h1tot)[1]:
    for j in i:
        basis_from_h1tot.append(j)
#print(basis_from_h1tot)

b_from_h1tot=spectrum_and_eigenspace_of_diagonal_matrix(h1tot)[1] #is a list of lists
basis_from_h1tot=spectrum_and_eigenspace_of_diagonal_matrix(h1tot)[2] #is a 2d array with shape (1,9)
#print("")
#print(basis_from_h1tot)
#print(basis_from_h1tot.shape)




# In[11]:


#print("spectrum of h2tot is :")
#print(spectrum_and_eigenspace_of_diagonal_matrix(h2tot)[0])
#print(" ")
#print("list of eigenstates of h2tot is :")
#print(spectrum_and_eigenspace_of_diagonal_matrix(h2tot)[1])

#basis_from_h2tot=[]
#for i in spectrum_and_eigenspace_of_diagonal_matrix(h2tot)[1]:
#    for j in i:
#        basis_from_h2tot.append(j)
#print(basis_from_h2tot)

b_from_h2tot=spectrum_and_eigenspace_of_diagonal_matrix(h2tot)[1] #is a list of lists
basis_from_h2tot=spectrum_and_eigenspace_of_diagonal_matrix(h2tot)[2] #is a 2d array with shape (1,9)
#print("")
#print(basis_from_h2tot)
#print(basis_from_h2tot.shape)


# Now that we have computed the eigenspaces of $h_{1_{tot}}$ and $h_{2_{tot}}$, by making the union of the basis of each basis of these eigenspaces, one obtains a basis in which the hamiltonian is block diagonal. In the next two cells, we will build the hamiltonian in this new basis, using a method that is not efficient ; we build the matrix in the new basis, coefficient by coefficient ; it is not efficient since (as you will see), there is a loop in a loop, and the total does $3^{2N}$ operations, which is a lot. 

# In[12]:


#def square_matrix_element(A,i,j):
#    '''Take a matrix square matrix A and return the matrix element <ei,A,ej>, where e denote is the 
#    canonical vector '''
#    #creation matrice avec 1 ligne et N colonnes (N=taille A), avec un seul élément non nul en position ligne 1 
#    #colonne j
#    ej=np.zeros((np.shape(A)[0],1))
#    ej[j]=1
#    Aj=np.dot(A,ej)
#    ei_transpose=np.zeros((1,(np.shape(A)[0])))
#    ei_transpose[0][i]=1
#    return np.dot(ei_transpose,Aj)[0][0]


# In[13]:


#matrix=np.zeros((3**number_of_sites,3**number_of_sites))
#for i in range(0,len(hamiltonien[0])):
#    for j in range(0,len(hamiltonien[0])):
#        matrix[i][j]=square_matrix_element(hamiltonien,basis_from_h1tot[0][i],basis_from_h1tot[0][j]).real
#plt.spy(matrix)
#print(matrix)


# To build the hamiltonian in the new basis, one can use an alternative way, that makes avantage of the fact that 
# the new basis is just a permutation of the canonical basis. To do this, we will use the numpy advanced indexing features (see https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing for the official documentation on advanced indexing). 

# Consider a square matrix $A\in M_N(\Bbb{C})$, where $N\in \Bbb{N}\setminus\{0\}$. In python, it is possible to access specific rows/columns of the matrix. Suppose that $A$ is a 2-dimentional array, and that $i\in \{ 0,1,...,N-1\} $.
# If $$\texttt{r=A[i,:]}$$, then $r$ is a 1-dimentional array of shape (1,) made of the elements of row number $i$ of $A$, put in the same order as in $A$.  If $$\texttt{c=A[:,j]}$$, then $c$ is a 1-dimentional array of shape (1,) made of the elements of row number $j$ of $A$, put in the same order as in $A$.

# In[14]:


# m=np.arange(16).reshape(4,4)
# print("m= ")
# print(m)
# row1_m=m[1,:]
# print(" ")
# print("row 1 of m is :")
# print(row1_m)
# col1_m=m[:,1]
# print(" ")
# print("column 1 of m is :")
# print(col1_m)


# Sometimes, we want to access sub-matrices of $A$. If $$\texttt{subA=A[i:j,k:l]}$$
# the $\texttt{subA}$ is a 2-dimentional array constructed as follows: take the rows $A[i],A[i+1],...,A[j-1]$ of $A$. Then $sub(A)$ is the 2-dimentional array made of these rows, but in which we only keep the column indices $k,k+1,...,l-1$. So in $sub(A)$, we will have $j-i$ rows, and in each row, $l-k$ columns.
# 
# Lets understand this through an example :

# In[15]:


# subm=m[0:2,0:3]
# print(subm)
# subm2=m[0:3,0:4]
# print(subm2)


# So we have seen that one can access column and row of a matrix, and also "slices" of matrices. Now we want to access in one command, several items of a matrix, independently, a put them in a new matrix. To do this, one can use the advanced indexing features of python.

# Consider $\texttt{col}$ and $\texttt{row}$ two arrays of shape $(N,)$. Consider 
# $$\texttt{l=A[row[:,np.newaxis],col]}$$
# Then $\texttt{l}$ is a 2-dimentional array (a matrix) of shape $(N,N)$ where element in row i, column j, is $\texttt{A[row[i],col[j]]}$. Lets consider the following example

# In[16]:


# A=np.arange(9).reshape(3,3)
# print(A)
# row=np.array([0,1,1])
# column=np.array([2,1,0])
# print(A[row[:,np.newaxis],column])
# for i in range(0,3):
#     for j in range(0,3):
#         print(A[row[:,np.newaxis],column][i,j]==A[row[i],column[j]])


# Now, consider $\texttt{col}$ and $\texttt{row}$ that are two arrays of shape $(N',)$ where $N'<N$, where remember that $N$ is the size of the matrix $A$. Then, if 
# $$\texttt{l=A[row[:,np.newaxis],column]}$$
# $\texttt{l}$ is a 2-dimentional array of shape $(N',N')$ where element in row i, column j, is $\texttt{A[row[i],col[j]]}$.

# In[17]:


# A=np.arange(9).reshape(3,3)
# print(A)
# row=np.array([0,1])
# column=np.array([2,1])
# print(A[row[:,np.newaxis],column])
# for i in range(0,2):
#     for j in range(0,2):
#         print(A[row[:,np.newaxis],column][i,j]==A[row[i],column[j]])


# In[18]:


#basis_from_h1tot=np.array(basis_from_h1tot) #convertion en array de la liste basis_from_h1tot
hamiltonien_in_h1basis=hamiltonien[basis_from_h1tot[0][:,None],basis_from_h1tot[0]] #ecriture de l'hamiltonien dans la nouvelle base
#print(hamiltonien_in_h1basis)
#plt.spy(hamiltonien_in_h1basis)


# To check if the result makes sense, one way is to see if the hamiltonien still commutes with $h_{1_{tot}}$, when 
# expressed in the basis of eigenstates of $h_{1_{tot}}$.

# In[19]:


#h1tot_in_h1basis=h1tot[basis_from_h1tot[0][:,None],basis_from_h1tot[0]]
#print(np.dot(hamiltonien_in_h1basis,h1tot_in_h1basis)-np.dot(h1tot_in_h1basis,hamiltonien_in_h1basis))


# In the next cell, we compute the hamiltonian in each block. 

# In[20]:


#for i in spectrum_and_eigenspace_of_diagonal_matrix(h1tot)[1]:
#    i=np.array(i) #convertion en array pour pouvoir utiliser le advanced indexing de numpy
#    block=hamiltonien[i[:,None],i]
#    print(block)
#    print("block")


# Pour chacun des blocs de la matrice de $\hat{H}$ dans la base $\texttt{basis_from_h1tot}$, il faut maintenant regrouper les états propres ayant la même valeur propre pour $h_{2_{tot}}$. L'objectif de la prochaine cellule est donc de créer un array $\texttt{b_from_h1h2tot}$ qui est un 2-dimentional array, ou chaque element est obtenue en prenant les élements de $\texttt{b_from_h1tot}$ et en effectuant leur intersection avec les élements de $\texttt{b_from_h2tot}$. Dans cette base, l'hamiltonien aura une forme block-diagonale, avec plus de block que dans la base $\texttt{basis_frol_h1tot}$. 
# 
# Diagonaliser la matrice de l'hamiltonien dans la base $\texttt{b_from_h1h2tot}$ est maintenant équivalent à diagonaliser chacun des blocks. Comme chaque block est une matrice réelle, est symétrique, on pourra utiliser la commande $\texttt{scipy.sparse.linalg.eigsh}$. 
# 
# 

# In[21]:


b_from_h1h2tot=[]
nb_blocks=0
for i in b_from_h1tot:
    permutation_block=[]
    for j in b_from_h2tot:
        i=np.array(i) #on converti en array les listes pour pouvoir utiliser np.intersect1d()
        j=np.array(j)
        inter=np.intersect1d(i,j)
        inter=list(inter) #on converti l'intersection en liste pour pouvoir l'ajouter à b_from_h1h2tot
        if inter !=[]:
            permutation_block.append(inter)
            nb_blocks+=1
    b_from_h1h2tot.append([item for sublist in permutation_block for item in sublist])
#print("le nombre de block de l'hamiltonien, en utilisant h1tot et h2tot, est : "+str(nb_blocks))
#print(b_from_h1h2tot)


# Dans la prochaine cellule, le spectre de l'hamiltonien (avec répétitions) est calculé, puis affiché. 

# In[22]:


import scipy.linalg as la


total_s=[]
compteur2=0
for i in b_from_h1h2tot:

    i=np.array(i)
    
    s,U=la.eigh(hamiltonien[i[:,None],i]) #s is the spectrum, U set of eigenvalues

    for j in s:
       
        total_s.append(j)
    
    compteur2+=1

total_s=sorted(total_s) #spectre de l'hamiltonien, trié dans l'ordre croissant
print(total_s)


# In the next cell, we write a code that returns the spectrum, and a list where each element is the multiplicity of
# the eigenvalue that has the index of the element.

# In[23]:


def truncate(number, digits) -> float:
    # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
    nbDecimals = len(str(number).split('.')[1]) 
    if nbDecimals <= digits:
        return number
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

spectrum=[total_s[0]] 
degen=[1]
com=0

for i in range(1,len(total_s)):
    if total_s[i]>total_s[i-1] and not math.isclose(total_s[i], total_s[i-1], abs_tol=1e-10):
        spectrum.append(truncate(total_s[i],3))
        com=1
        degen.append(com)
    else:
        degen[len(degen)-1]+=1
print("spectrum =")
print(spectrum)
print("with deg.")
print(degen)
print("")
degen_s=sorted(degen)
mult_decomp=[]
mult_decomp.append(degen_s.count(degen[0]))
mult_decomp.append(degen_s[0])
for i in degen_s:
    if i > mult_decomp[len(mult_decomp)-1]:
        mult_decomp.append(degen_s.count(i))
        mult_decomp.append(i)
#print(mult_decomp)
print("")

print(datetime.now() - start)
