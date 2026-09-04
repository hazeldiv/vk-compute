#ifndef prune_h
#define prune_h

#include "model.h"

#define PRUNED_VOCAB_DIR "../pruned-vocab"

int pruneVocab(const char* modelDir, const model_config* spec);

#endif
