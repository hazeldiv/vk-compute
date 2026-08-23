#include <stdio.h>
#include <string.h>

void compute(void);
void validation(void);

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "val") == 0) {
        validation();
    } else {
        compute();
    }
    return 0;
}
