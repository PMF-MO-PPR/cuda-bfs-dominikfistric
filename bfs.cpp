#include "bfs.h"
#include "sparse.h"

void find_path(CSCMat const & incidence, int dst, std::vector<int> const & level, std::vector<int> & path)
{
    if (level[dst] == -1) {
        std::cout << "NE POSTOJI PUT!" << std::endl;
        return;
    }

    while (level[dst] > 0) {
        for (int i = incidence.colPtrs[dst]; i < incidence.colPtrs[dst + 1]; ++i) {
            int y = incidence.rowIdx[i];
            if (level[y] != level[dst] - 1) continue;
            path.push_back(dst);
            dst = y;
            break;
        }
    }
    path.push_back(dst);
}
