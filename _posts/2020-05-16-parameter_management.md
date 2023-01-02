---
layout: post
title: Hyper-parameter Management in Deep Learning Projects
date: 2020-05-16T00:45:00+04:30
tags: deep_learning programming
categories: programming
---

Deep Learning (DL) projects have a plethora of hyper-parameters. Especially in research, they are like knobs that should be adjusted to yields the best results. There are multiple ways one can manage these hyper-parameters in Python. In this article, I'm going to discuss 3 common approaches.

# Argument Parser
The first approach which is very easy and common is to use Python's built-in argument parser module. All we need to do is to define some parameters preferably with default values or pass/change them during the program execution. Let's create a simple script called `main.py`:

```python
# import the module
import argparse


if __name__ == "__main__":
    # create a parser object
    parser = argparse.ArgumentParser()
    # define some arguments
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)

    # access the define arguments in project
    args = parser.parse_args()
    print(args.batch_size)
    print(args.num_epochs)
```

When experimenting with different values for hyper-parameters, we can change the value of each argument when running the script:

```bash
python main.py --batch_size 32 --num_epochs 100
```

Arguments that have a double dash (`--`) at the beginning of their name are optional. If you don't put `--` at the beginning of the argument name, it means that the argument is positional and should be provided when executing the script. You can read more about `argparse` module at [official docs](https://docs.python.org/3/library/argparse.html).

This approach is effortless to implement and can be easily changed in experiments with different values. But, it has some downsides too. If the project has a lot of hyper-parameters to handle, which in research, it is usually the case, this list can become extensive. One way to circumvent this issue is to create a separate python file to hold and parse these parameters. Then, the only thing that needs to be done, is to change the default value when you want to experiment with other  values.

# Using Auxiliary File Formats
We can also use an extra file (with different extension) to manage the hyper-parameters. Here, I'm going to introduce 3 different common files to store hyper-parameters.

## `.ini` File
Python has a built-in module to parse `.ini` files, therefore using `.ini.` file to store parameters in Python is easy. Also, in `.ini` files we can group hyper-parameters based on their usage. Let's create a `config.ini` file:
```ini
[TRAINING]
epochs = 200
learning_rate = 1e-4

[DIRECTORIES]
root = ./
save_model = ./save
log = ./logs
```
Here, we separated the parameters into two groups, one is related to training the model, and the other is related to managing directories. We need a helper function to help us parse the `config.ini` file.

```python
# import the required module
import configparser


def config_parser(module_name: str=None) -> Dict[str, str]:
    """A helper function which receive the config category name,
       and returns a dictionary containing the values of 
       hyper-parameters of the specific category
    """
    # create a config parser object
    config = configparser.ConfigParser()
    # pass it the absolute address of config.ini file
    config.read("config.ini")
    try:
        # if the key present in the config file, return it
        return config[module_name]
    except KeyError as err:
        print(f"Module name should be one of the:\n "
              f"{config.sections()} not {err}")
```

Now that we have a helper function to parse the config file, we can use it wherever we want in the main script to obtain required hyper-parameters:

```python
if __name__ == "__main__":
    DIRECTORIES = config_parser("DIRECTORIES")
    TRAINING = config_parser("TRAINING")

    # treat the files like dictionary
    print(DIRECTORIES["root"])
    print(DIRECTORIES["log"])
    print(int(TRAINING["epochs"]))
    print(float(TRAINING["learning_rate"]))
```
One thing that should be noted here is that the config parser treats each value as a string, therefore, we need to convert it to the desired data type. And it only supports one level deep hierarchy. But the bright side is that it keeps your main script cleaner and shorter.

## Python Script as a Config Manager
We can also use an auxiliary python file to hold hyper-parameters. Create a file called `configs.py` containing only a dictionary:
```python
config = {
    "epoch": 200,
    "learning_rate": 1e-4,
    "root": "./"
}
```
We can add all model hyper-parameters as a dictionary. For different parts of the model we can create different dictionaries and use them as follows:
```python
from configs import config


if __name__ == "__main__":
    print(config["epoch"], type(config["epoch"]))
    print(config["root"], type(config["root"]))
    print(config["learning_rate"], type(config["learning_rate"]))
```
In contrast to `.ini` file, the data types are reserved and there is no need for conversion. As easy as this approach is, but it is not very common. The reason for this is that it is not language agnostic.

## YAML Files
YAML file format is increasingly becoming more popular due to readability and conciseness. There are multiple third-party libraries to handle YAML files in Python. Create `config.yaml` file:
```yaml
Datasets:
  # flip dataset consists of 256 images
  flip-dataset:
    batch: 32
    shuffle: True
  # flot dataset consists of 1024 images
  flop-dataset:
    batch: 64
    shuffle: False
    
Directories:
  root: "./"
  logs: "logs/"
```

As we can see, it supports multi-level hierarchy configs and even comments! In the above case, we have two different datasets, `flip-dataset` and `flop-dataset` with different configurations. To parse this file, we need a third-party library such as [_OmegaConf_](https://omegaconf.readthedocs.io/en/latest/index.html), [_PyYAML_](https://pyyaml.org) or [_Confuse_](https://confuse.readthedocs.io/en/latest/). Here, I'm going to use PyYAML. Install it using your favorite Python package manager:
```bash
pip install --user pyyaml
```
We can use it in our project as follow:
```python
import yaml


if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config["Datasets"]["flip-dataset"]["batch"])
    print(config["Datasets"]["flip-dataset"]["shuffle"])
    print(config["Datasets"]["flop-dataset"]["batch"])
    print(config["Directories"]["logs"])
```
The good thing is that it also reserves data types and is language agnostic. One benefit of [_OmegaConf_](https://omegaconf.readthedocs.io/en/latest/index.html) with respect to other libraries is that the parameters are treated as attributes and we can call them like attributes of a class by using `.` notation. One think that should be kept in mind is that, if we want using attribute calling notation, we should be careful with our naming. For example in the config above, name `flip-dataset` is not a valid python variable name (why?). Instead use underscores like `flip_dataset`.
```python
from omegaconf import OmegaConf


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")

    print(config.Datasets)
    print(config.Directories.logs)
```

You can use different file formats of your choice such as `JSON` and `XML` to declutter the project codebase. But managing nested configs with such formats is exasperating.

# Hydra
In situations where we have a huge project with a plethora of hyper-parameters, multiple datasets, multiple optimizer and we want to experiment with different values but we are short on time, [__Hydra__](https://hydra.cc) is the hero we want. It is "a framework for elegantly configuring complex applications" developed at Facebook. It uses YAML, therefore inherent its advantages. It also gives us the ability to run multiple experiments in parallel with different configurations. We can put all our configuration files in a separate directory such as `configs`. We can also create different folders to hold different configurations (called `config group`) for different parts of the project to prevent extensive config files and easily alternate between them. Another cool feature of _hydra_ is that you can treat YAML keys as arguments and change them when executing the program like `argparse` module. If you see the need to use _hydra_ in your project, its [tutorial page](https://hydra.cc/docs/tutorial/simple_cli/) is comprehensive. It is a feature-packed framework and writing a tutorial about it is out of scope of this short article.


# Wrapping up
Although it seems that the approaches above are incrementally get better and better, but actually, it is not the case. All of these approaches have their own pros and cons. Choosing one is highly correlated to your project's size and needs. I hope by showing simple examples I could give you an insight on how to choose an approach. So, which one is your favorite approach?
