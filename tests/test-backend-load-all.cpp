#include "ggml-backend.h"

#include <cstdio>

static bool backend_registered(const char * name) {
    return ggml_backend_reg_by_name(name) != nullptr;
}

static int count_missing_backends(int argc, char ** argv) {
    int missing = 0;
    for (int i = 2; i < argc; ++i) {
        if (backend_registered(argv[i])) {
            continue;
        }
        fprintf(stderr, "backend '%s' is not registered after loading from '%s'\n", argv[i], argv[1]);
        ++missing;
    }
    return missing;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <backends-dir> <backend-name>...\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all_from_path(argv[1]);
    return count_missing_backends(argc, argv) == 0 ? 0 : 1;
}
