#podaci su prikazani u datoteci u kojoj je svaki redak jedno slovo, u formatu:
#1. broj (broj od 0 do 25 - predstavlja slovo)
#2. 784 broja(slovo je slika velicine 28x28 piksela) od 0 do 255 koji predstavljaju nijansu boje (ako je 0, onda je taj piksel bijel, 
# odnosno nista nije upisano, ako je 255, piksel je potpuno crn)

#funkcija koja stvara oznake i značajke
def stvori_oznake_i_znacajke(podaci):

	#prvi podatak u retku je točna oznaka slova koje se treba prepoznati
	oznake = podaci[0]

	#značajke su ostali podaci koji služe za učenje mreže (boje piksela)
	znacajke = podaci.loc[:,1:784]
	#skaliramo podatke u segment [0, 1] da bi s njima bilo lakše raditi
	znacajke = znacajke / 255
	#funkcija vraća oznake i značajke
	return znacajke, oznake
