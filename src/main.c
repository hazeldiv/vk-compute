#include <stdio.h>
#include <string.h>
#include "dispatch.h"

void compute(int argc, char** argv);
void validation(void);

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "val") == 0) {
        validation();
    } else {
        setTimingEnabled(argc > 1 && strcmp(argv[1], "log") == 0);
        compute(argc, argv);
    }
    return 0;
}
