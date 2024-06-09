#include <iostream>
#include <string>
#include "bfs.h"
#include "sparse.h"
#include "labirint_io.h"

void check_input(LabIOMatrix const & mat, int start_row, int start_col,
	             int stop_row, int stop_col);

///////// Vaša CUDA jezgra dolazi ovdje ////////////////////


VAŠ KOD



////////////////////////////////////////////////////////////

int main(int argc, char * argv[])
{
	int start_row = -1; // polazna točka row
	int start_col = -1; // polazna točka col
	int stop_row = -1;  // završna točka row
	int stop_col = -1;  // završna točka col
	std::string file_name = "labirint.txt"; // ulazna datoteka s labirintom
    
	if(argc >= 6){
		start_row = std::stoi(argv[1]);
		start_col = std::stoi(argv[2]);
		stop_row = std::stoi(argv[3]);
		stop_col = std::stoi(argv[4]);
		file_name = argv[5];
	}
	else{
		std::cerr << "Upotreba: " << argv[0] << " start_row start_col stop_row stop_col file_name\n";
		std::cerr << "Brojevi stupaca i redaka idu od nule.\n";
		std::exit(1);
	}

	// Kreiraj labirint. Labirint je zadan s matricom tipa LabMatrix.
	LabIOMatrix mat;
    mat.read(file_name);
    check_input(mat, start_row, start_col, stop_row, stop_col);

	// Kreiraj graf iz labirinta. Funkcija vraća matricu incidencije koja je ovdje dana kao 
	// puna matrica. 
	IncidenceMat incidence(mat);  
	CSRMat csr_incidence(incidence);
	CSCMat csc_incidence(incidence);

//	  csr_incidence.print();
//    csc_incidence.print();
	
	int start_idx = mat(start_row, start_col);
	int stop_idx  = mat(stop_row,stop_col);
	std::cout << "start index = " << start_idx << ", stop index = " << stop_idx << "\n";

    /// VAŠ CUDA kod  DOLAZI OVDJE /////////////////////////////////////////
    // ALOCIRAJ MEMORIJU NA GPU, KOPIRAJ PODATKE S CPU NA GPU, 
    // POZOVI JEZGRU, KOPIRAJ LEVEL POLJE S GPU NA CPU.
   
       VAŠ KOD
    ///////////////////////////////////////////////////////////////////////

    std::vector<int>  path;  // STAZA
    // IZRAČUNAJ STAZU
	find_path(csc_incidence, stop_idx, level, path); 
    // PRINTAJ STAZU U DATOTEKU
    mat.print_ascii("out_"+base_name(file_name), path);

   // POČISTITE MEMORIJU //////////////////// 
     VAŠ KOD
    ///////////////////////////////////////////
    return 0;
}
