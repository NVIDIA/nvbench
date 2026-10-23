file_version = (1, 1, 0)

file_version_string = "{}.{}.{}".format(
    file_version[0], file_version[1], file_version[2]
)

# Shared result-processing contract, separate from the JSON schema version.
# Keep in sync with json_printer::get_scripts_version().
script_version = (1, 0, 0)
script_version_string = ".".join(map(str, script_version))


def check_script_version(filename, root_node):
    try:
        version_node = root_node["meta"]["version"]["scripts"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{filename}: missing script compatibility version (meta.version.scripts); "
            "regenerate the benchmark results"
        ) from exc
    actual = version_node.get("string") if isinstance(version_node, dict) else None
    if actual != script_version_string:
        raise ValueError(
            f"{filename}: unsupported script compatibility version {actual!r}; "
            f"expected {script_version_string!r}"
        )


def check_file_version(filename, root_node):
    try:
        version_node = root_node["meta"]["version"]["json"]
    except KeyError:
        print("WARNING:")
        print("  {} is written in an older, unversioned format. ".format(filename))
        print("  It may not read correctly.")
        print("  Reader expects JSON file version {}.".format(file_version_string))
        return

    # TODO We could do something fancy here using semantic versioning, but
    # for now just warn on mismatch.
    # 1.1 adds optional timer metadata; the 1.0 format remains readable.
    if version_node["string"] not in ("1.0.0", file_version_string):
        print("WARNING:")
        print(
            "  {} was written using a different NVBench JSON file version.".format(
                filename
            )
        )
        print("  It may not read correctly.")
        print(
            "  (file version: {} reader version: {})".format(
                version_node["string"], file_version_string
            )
        )
