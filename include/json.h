#ifndef json_h
#define json_h

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type;

typedef struct json_value {
    json_type type;
    int boolean;
    double number;
    char* string;
    struct json_value* items;
    struct json_value* values;
    char** keys;
    int count;
} json_value;

json_value* json_parse_file(const char* path);
json_value* json_get(const json_value* v, const char* key);
const char* json_get_str(const json_value* v, const char* key, const char* def);
double json_get_num(const json_value* v, const char* key, double def);
int json_get_int(const json_value* v, const char* key, int def);
int json_get_bool(const json_value* v, const char* key, int def);
void json_free(json_value* v);

#endif
