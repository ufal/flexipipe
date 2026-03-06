/*
 * Minimal C launcher for flexipipe. Replaces the bash wrapper so we only
 * start one Python process (no "check" invocation). Saves ~0.5–2s on startup.
 *
 * Build: cc -O2 -o flexipipe_launcher flexipipe_launcher.c
 * Use:   ./flexipipe_launcher [args...]  →  exec python3 -m flexipipe [args...]
 *
 * Environment:
 *   VENV_PATH       → use $VENV_PATH/bin/python
 *   VIRTUAL_ENV     → use $VIRTUAL_ENV/bin/python
 *   FLEXIPIPE_REPO_PATH → set PYTHONPATH to this before exec (dev install)
 *
 * If a file <dirname(executable)/flexipipe.venv> exists (written at install time
 * when run from a venv), its first line is used as VENV_PATH so the launcher
 * uses the same Python that was used to install the wrapper.
 */

#define _POSIX_C_SOURCE 200809L
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

static const char *const MODULE = "flexipipe";
static const char *const PYTHON_NAMES[] = { "python3", "python", NULL };
static const char *const VENV_FILE = "flexipipe.venv";

static int exists(const char *path) {
    return access(path, X_OK) == 0;
}

/* Resolve path to this executable; write into buf, return 0 on success. */
static int get_executable_path(char *buf, size_t size) {
    if (size == 0) return -1;
    buf[0] = '\0';
#ifdef __linux__
    ssize_t n = readlink("/proc/self/exe", buf, size - 1);
    if (n <= 0) return -1;
    buf[n] = '\0';
    return 0;
#elif defined(__APPLE__)
    uint32_t len = (uint32_t)size;
    if (_NSGetExecutablePath(buf, &len) != 0) return -1;
    return 0;
#else
    (void)buf;
    (void)size;
    return -1;
#endif
}

/* If flexipipe.venv exists next to the executable, set VENV_PATH from its first line. */
static void maybe_load_venv_file(void) {
    if (getenv("VENV_PATH") != NULL) return;  /* Already set */
    char exe[PATH_MAX];
    if (get_executable_path(exe, sizeof(exe)) != 0) return;
    char *slash = strrchr(exe, '/');
    if (!slash) return;
    *slash = '\0';  /* dirname in place */
    char venv_path[PATH_MAX];
    snprintf(venv_path, sizeof(venv_path), "%s/%s", exe, VENV_FILE);
    FILE *f = fopen(venv_path, "r");
    if (!f) return;
    char line[PATH_MAX];
    if (fgets(line, (int)sizeof(line), f)) {
        size_t n = strcspn(line, "\r\n");
        line[n] = '\0';
        if (n > 0) setenv("VENV_PATH", line, 0);  /* 0 = don't overwrite existing */
    }
    fclose(f);
}

/* Find Python: VENV_PATH (env or from .venv file), then VIRTUAL_ENV, then PATH (python3, python). */
static char *find_python(char *buf, size_t size) {
    maybe_load_venv_file();
    const char *venv = getenv("VENV_PATH");
    if (venv && venv[0]) {
        snprintf(buf, size, "%s/bin/python", venv);
        if (exists(buf)) return buf;
    }
    venv = getenv("VIRTUAL_ENV");
    if (venv && venv[0]) {
        snprintf(buf, size, "%s/bin/python", venv);
        if (exists(buf)) return buf;
    }
    const char *path_env = getenv("PATH");
    if (!path_env) path_env = "/usr/bin:/bin";
    for (const char *const *p = PYTHON_NAMES; *p; p++) {
        char path_copy[4096];
        snprintf(path_copy, sizeof(path_copy), "%s", path_env);
        for (char *dir = strtok(path_copy, ":"); dir; dir = strtok(NULL, ":")) {
            snprintf(buf, size, "%s/%s", dir, *p);
            if (exists(buf)) return buf;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    char python_path[PATH_MAX];
    if (!find_python(python_path, sizeof(python_path))) {
        fprintf(stderr, "flexipipe: Python not found. Set VENV_PATH or ensure python3 is on PATH.\n");
        return 127;
    }

    /* Optional: dev install via FLEXIPIPE_REPO_PATH */
    const char *repo = getenv("FLEXIPIPE_REPO_PATH");
    if (repo && repo[0]) {
        char *old = getenv("PYTHONPATH");
        char newpath[4096];
        if (old && old[0])
            snprintf(newpath, sizeof(newpath), "%s:%s", repo, old);
        else
            snprintf(newpath, sizeof(newpath), "%s", repo);
        setenv("PYTHONPATH", newpath, 1);
    }

    /* argv: [python_path, "-m", MODULE, argv[1], argv[2], ...] */
    int n = argc + 2;
    char **new_argv = calloc((size_t)n + 1, sizeof(char *));
    if (!new_argv) { perror("flexipipe"); return 126; }
    new_argv[0] = python_path;
    new_argv[1] = (char *)"-m";
    new_argv[2] = (char *)MODULE;
    for (int i = 1; i < argc; i++)
        new_argv[2 + i] = argv[i];
    new_argv[2 + argc] = NULL;

    execv(python_path, new_argv);
    perror("flexipipe");
    free(new_argv);
    return 126;
}
