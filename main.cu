#include <iostream>
#include <string>
#include "bfs.h"
#include "sparse.h"
#include "labirint_io.h"

void check_input(LabIOMatrix const & mat, int start_row, int start_col,
                 int stop_row, int stop_col);

///////// Vaša CUDA jezgra dolazi ovdje ////////////////////

__global__
void bfs_kernel(CSRMat *incidence, int *level, int *prev_front, int *curr_front,
                const int *prev_front_size, int *curr_front_size, int currentLevel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < *prev_front_size) {
        int vertex = prev_front[i];
        if (level[vertex] == currentLevel - 1) {
            for (int edge = incidence->rowPtrs[vertex]; edge < incidence->rowPtrs[vertex + 1]; ++edge) {
                int neighbor = incidence->colIdx[edge];
                if (atomicCAS(&level[neighbor], -1, currentLevel) == -1) {
                    int idx = atomicAdd(curr_front_size, 1);
                    curr_front[idx] = neighbor;
                }
            }
        }
    }
}
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

    // csr_incidence.print();
    // csc_incidence.print();

    int start_idx = mat(start_row, start_col);
    int stop_idx  = mat(stop_row,stop_col);
    std::cout << "start index = " << start_idx << ", stop index = " << stop_idx << "\n";

    /// VAŠ CUDA kod  DOLAZI OVDJE /////////////////////////////////////////
    // ALOCIRAJ MEMORIJU NA GPU, KOPIRAJ PODATKE S CPU NA GPU,
    // POZOVI JEZGRU, KOPIRAJ LEVEL POLJE S GPU NA CPU.

    const int NODE_COUNT = mat.no_blocks();

    CSRMat *d_csr_incidence;
    cudaMalloc((void**) (&d_csr_incidence), sizeof(CSRMat));
    cudaMemcpy(d_csr_incidence, &csr_incidence, sizeof(CSRMat), cudaMemcpyHostToDevice);

    int *d_rowPtrs, *d_colIdx;
    cudaMalloc(&d_rowPtrs, (csr_incidence.nrows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, csr_incidence.nelem * sizeof(int));

    cudaMemcpy(d_rowPtrs, csr_incidence.rowPtrs, (csr_incidence.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, csr_incidence.colIdx, (csr_incidence.nelem) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(&(d_csr_incidence->rowPtrs), &d_rowPtrs, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_csr_incidence->colIdx), &d_colIdx, sizeof(int*), cudaMemcpyHostToDevice);

    std::vector<int> level(csr_incidence.nrows, -1);
    level[start_idx] = 0;

    int *d_level;
    cudaMalloc(&d_level, level.size() * sizeof(int));
    cudaMemcpy(d_level, level.data(), level.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_prev_visited;
    cudaMalloc(&d_prev_visited, sizeof(int) * NODE_COUNT);
    cudaMemcpy(d_prev_visited, &start_idx, sizeof(int), cudaMemcpyHostToDevice);
    int *d_prev_visited_count;
    cudaMalloc(&d_prev_visited_count, sizeof(int));
    int prev_visited_count = 1;
    cudaMemcpy(d_prev_visited_count, &prev_visited_count, sizeof(int), cudaMemcpyHostToDevice);

    int *d_curr_visited;
    cudaMalloc(&d_curr_visited, sizeof(int) * NODE_COUNT);
    int *d_curr_visited_count;
    cudaMalloc(&d_curr_visited_count, sizeof(int));

    int current_level = 1;

    while (prev_visited_count > 0) {
        const int BLOCK = 128;
        const int GRID = (prev_visited_count + BLOCK - 1) / BLOCK;
        bfs_kernel<<<BLOCK, GRID>>>(d_csr_incidence, d_level, d_prev_visited,
                                    d_curr_visited, d_prev_visited_count,
                                    d_curr_visited_count, current_level++);
        cudaDeviceSynchronize();
        std::swap(d_prev_visited, d_curr_visited);
        std::swap(d_prev_visited_count, d_curr_visited_count);
        cudaMemcpy(&prev_visited_count, d_prev_visited_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(d_curr_visited_count, 0, sizeof(int));
    }

    cudaMemcpy(level.data(), d_level, level.size() * sizeof(int), cudaMemcpyDeviceToHost);

    ///////////////////////////////////////////////////////////////////////


    std::vector<int>  path;  // STAZA
    // IZRAČUNAJ STAZU
    find_path(csc_incidence, stop_idx, level, path);
    // PRINTAJ STAZU U DATOTEKU
    mat.print_ascii("out_"+base_name(file_name), path);

    // POČISTITE MEMORIJU ////////////////////
    cudaFree(d_prev_visited);
    cudaFree(d_prev_visited_count);
    cudaFree(d_curr_visited);
    cudaFree(d_curr_visited_count);
    cudaFree(d_level);
    cudaFree(d_rowPtrs);
    cudaFree(d_colIdx);
    cudaFree(d_csr_incidence);
    ///////////////////////////////////////////

    return 0;
}
