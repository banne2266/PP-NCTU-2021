#include <cstdio>
#include <vector>
using namespace std;



int main () {
    int n, m, l;
    int *a_mat, *b_mat;
    scanf("%d %d %d", &n, &m, &l);
    vector<vector<int>> a(n, vector<int>(m, 0));
    vector<vector<int>> b(m, vector<int>(l, 0));
    vector<vector<int>> c(n, vector<int>(l, 0));

    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            scanf("%d", &a[i][j]);
        }
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < l; j++){
            scanf("%d", &b[i][j]);
        }
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < l; j++){
            int temp = 0;
            for(int k = 0; k < m; k++){
                temp += a[i][k] * b[k][j];
            }
            c[i][j] = temp;
        }
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < l; j++){
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    return 0;
}