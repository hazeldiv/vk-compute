#include <stdio.h>
#include <string.h>
#include "dispatch.h"

void serverMain(int argc, char** argv);
void validation(void);
void memInfo(void);

int main(int argc, char** argv) {
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc > 1 && strcmp(argv[1], "val") == 0) {
        validation();
    } else if (argc > 1 && strcmp(argv[1], "meminfo") == 0) {
        memInfo();
    } else {
        serverMain(argc, argv);
    }
    return 0;
}
