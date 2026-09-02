#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

typedef struct {
    const char* p;
    const char* end;
} jctx;

static void jws(jctx* c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\n' || *c->p == '\r' || *c->p == '\t')) c->p++;
}

static int jch(jctx* c) {
    return c->p < c->end ? *c->p : '\0';
}

static char* jread_string(jctx* c) {
    if (jch(c) != '"') return NULL;
    c->p++;
    size_t cap = 32;
    size_t n = 0;
    char* out = (char*)malloc(cap);
    while (c->p < c->end) {
        char ch = *c->p;
        if (ch == '\\') {
            char esc = (c->p + 1 < c->end) ? c->p[1] : '\0';
            c->p += (c->p + 1 < c->end) ? 2 : 1;
            char dec = esc;
            if (esc == 'n') dec = '\n';
            else if (esc == 't') dec = '\t';
            else if (esc == 'r') dec = '\r';
            if (n + 1 >= cap) {
                cap *= 2;
                out = (char*)realloc(out, cap);
            }
            out[n++] = dec;
            continue;
        }
        c->p++;
        if (ch == '"') break;
        if (n + 1 >= cap) {
            cap *= 2;
            out = (char*)realloc(out, cap);
        }
        out[n++] = ch;
    }
    out[n] = '\0';
    return out;
}

static json_value* jnew(json_type t) {
    json_value* v = (json_value*)calloc(1, sizeof(json_value));
    v->type = t;
    return v;
}

static void jpush_item(json_value* arr, json_value* item) {
    arr->items = (json_value*)realloc(arr->items, sizeof(json_value) * (size_t)(arr->count + 1));
    arr->items[arr->count++] = *item;
    free(item);
}

static void jpush_key(json_value* obj, char* key, json_value* val) {
    obj->keys = (char**)realloc(obj->keys, sizeof(char*) * (size_t)(obj->count + 1));
    obj->values = (json_value*)realloc(obj->values, sizeof(json_value) * (size_t)(obj->count + 1));
    obj->keys[obj->count] = key;
    obj->values[obj->count] = *val;
    free(val);
    obj->count++;
}

static json_value* jparse_value(jctx* c);

static json_value* jparse_object(jctx* c) {
    json_value* obj = jnew(JSON_OBJECT);
    c->p++;
    while (1) {
        jws(c);
        if (jch(c) == '}') { c->p++; return obj; }
        char* key = jread_string(c);
        if (!key) { json_free(obj); return NULL; }
        jws(c);
        if (jch(c) == ':') c->p++;
        jws(c);
        json_value* val = jparse_value(c);
        if (!val) { free(key); json_free(obj); return NULL; }
        jpush_key(obj, key, val);
        jws(c);
        if (jch(c) == ',') { c->p++; continue; }
        if (jch(c) == '}') { c->p++; return obj; }
        free(key);
        json_free(obj);
        return NULL;
    }
}

static json_value* jparse_array(jctx* c) {
    json_value* arr = jnew(JSON_ARRAY);
    c->p++;
    while (1) {
        jws(c);
        if (jch(c) == ']') { c->p++; return arr; }
        json_value* val = jparse_value(c);
        if (!val) { json_free(arr); return NULL; }
        jpush_item(arr, val);
        jws(c);
        if (jch(c) == ',') { c->p++; continue; }
        if (jch(c) == ']') { c->p++; return arr; }
        json_free(arr);
        return NULL;
    }
}

static json_value* jparse_value(jctx* c) {
    jws(c);
    char ch = jch(c);
    if (ch == '{') return jparse_object(c);
    if (ch == '[') return jparse_array(c);
    if (ch == '"') {
        json_value* v = jnew(JSON_STRING);
        v->string = jread_string(c);
        if (!v->string) { free(v); return NULL; }
        return v;
    }
    if (ch == 'n') {
        c->p += 4;
        return jnew(JSON_NULL);
    }
    if (ch == 't') {
        c->p += 4;
        json_value* v = jnew(JSON_BOOL);
        v->boolean = 1;
        return v;
    }
    if (ch == 'f') {
        c->p += 5;
        json_value* v = jnew(JSON_BOOL);
        v->boolean = 0;
        return v;
    }
    {
        char* endp = NULL;
        double num = strtod(c->p, &endp);
        if (endp == c->p) return NULL;
        c->p = endp;
        json_value* v = jnew(JSON_NUMBER);
        v->number = num;
        return v;
    }
}

json_value* json_parse_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[len] = '\0';
    jctx c;
    c.p = buf;
    c.end = buf + len;
    jws(&c);
    json_value* v = jparse_value(&c);
    free(buf);
    return v;
}

json_value* json_get(const json_value* v, const char* key) {
    if (v == NULL || v->type != JSON_OBJECT) return NULL;
    for (int i = 0; i < v->count; i++) {
        if (strcmp(v->keys[i], key) == 0) return &v->values[i];
    }
    return NULL;
}
const char* json_get_str(const json_value* v, const char* key, const char* def) {
    json_value* item = json_get(v, key);
    if (item == NULL || item->type != JSON_STRING) return def;
    return item->string;
}

double json_get_num(const json_value* v, const char* key, double def) {
    json_value* item = json_get(v, key);
    if (item == NULL || item->type != JSON_NUMBER) return def;
    return item->number;
}

int json_get_int(const json_value* v, const char* key, int def) {
    return (int)json_get_num(v, key, (double)def);
}

int json_get_bool(const json_value* v, const char* key, int def) {
    json_value* item = json_get(v, key);
    if (item == NULL) return def;
    if (item->type == JSON_BOOL) return item->boolean;
    if (item->type == JSON_NUMBER) return item->number != 0.0;
    return def;
}

void json_free(json_value* v) {
    if (v == NULL) return;
    if (v->type == JSON_STRING) free(v->string);
    if (v->type == JSON_ARRAY && v->items != NULL) {
        for (int i = 0; i < v->count; i++) json_free(&v->items[i]);
        free(v->items);
    }
    if (v->type == JSON_OBJECT && v->values != NULL) {
        for (int i = 0; i < v->count; i++) {
            json_free(&v->values[i]);
        }
        free(v->values);
        for (int i = 0; i < v->count; i++) free(v->keys[i]);
        free(v->keys);
    }
    memset(v, 0, sizeof(json_value));
}
