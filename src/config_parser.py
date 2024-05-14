import json 

def parse_config_file(arguments): 

    # Remove all arguments with NoneType value
    arguments_dict = vars(arguments)
    all_keys = list(arguments_dict.keys())

    for key in all_keys:
        if arguments_dict[key] == None: 
            del arguments_dict[key]

    with open(arguments.config) as file: 
        content = json.load(file)
        content.update(arguments_dict)
        arguments.__dict__ = content

    return arguments