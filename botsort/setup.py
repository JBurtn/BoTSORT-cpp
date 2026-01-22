import os
import subprocess
from pybind11 import get_cmake_dir, get_include
from pybind11.setup_helpers import Pybind11Extension
from glob import glob
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)
os.environ['CPPFLAGS'] = "-Wno-deprecated-enum-enum-conversion "\
                         "-flto=8 "\
                         "-O3 "\
                         "-march=native " ## Change this if it errors out with cloud compute use.
ext_modules = [
    Pybind11Extension(
        "botsort._botsort",
        sorted(glob("csrc/*.cpp")),
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=[
            get_include(),
            #'/usr/include',
            '/usr/include/eigen3',
            #'/usr/local/include',
            '/usr/include/opencv4'
        ],
        extra_compile_args=[],
        extra_link_args=subprocess.check_output(["pkg-config", "--libs", "opencv4"]).decode().split(),
        cxx_std=20,
    ),
]

setup(
    name="botsort",
    version=__version__,
    author="JBurtn",
    author_email="Jmburton7@gmail.com",
    description="A pybind11 variant of BotSort-Cpp (https://github.com/viplix3/BoTSORT-cpp)",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    packages=['botsort'],
    #cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)