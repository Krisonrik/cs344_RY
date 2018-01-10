|Linux|Windows|cpplint|cmakelint|
|---|---|---|---|
|[![Build Status](http://54.68.121.138:8080/buildStatus/icon?job=hvr-sample-build-unit-test-linux)](http://54.68.121.138:8080/job/hvr-sample-build-unit-test-linux/)|[![Build Status](http://54.68.121.138:8080/buildStatus/icon?job=hvr-sample-build-unit-test-windows)](http://54.68.121.138:8080/job/hvr-sample-build-unit-test-windows/)|[![Build Status](http://54.68.121.138:8080/buildStatus/icon?job=hvr-sample-cpplint)](http://54.68.121.138:8080/job/hvr-sample-cpplint/)|[![Build Status](http://54.68.121.138:8080/buildStatus/icon?job=hvr-sample-cmakelint)](http://54.68.121.138:8080/job/hvr-sample-cmakelint/)|

How to compile:
==============

**For LINUX**

~~~~~~~~~~~~~~~~~~~~~
    mkdir build

    cd build

    cmake ..

    make -j
~~~~~~~~~~~~~~~~~~~~~

**For WINDOWS**

~~~~~~~~~~~~~~~~~~~~~
    mkdir build

    cd build

    cmake .. -A x64 -C ..\initcmake.txt ..
~~~~~~~~~~~~~~~~~~~~~

- Need to specify the compiler otherwise it will be defaulted to the
  32 bit compiler

- Need to specify the initial cache file

- click on the .sln

- compile Ctrl+Shift+B on visual studio

**For Mac/Xcode**

~~~~~~~~~~~~~~~~~~~~~
    mkdir build

    cd build

    cmake .. -G Xcode
~~~~~~~~~~~~~~~~~~~~~

<br />

Setup PPA - For Linux
==============

~~~~~~~~~~~~~~~~~~~~~
    sudo sh -c "echo \"deb [arch=amd64 trusted=yes] http://52.33.197.63 debian/\" >> /etc/apt/sources.list"

    sudo sh -c "echo Package: \* >> /etc/apt/preferences"

    sudo sh -c "echo Pin: origin 52.33.197.63 >> /etc/apt/preferences"

    sudo sh -c "echo Pin-Priority: 1001 >> /etc/apt/preferences"

    sudo apt-get update
~~~~~~~~~~~~~~~~~~~~~

<br />

Setup Vcpkg - For Windows
==============

 - Clone vcpkg repository, then run bootstrap

~~~~~~~~~~~~~~~~~~~~~
    git clone https://github.com/Microsoft/vcpkg.git

    bootstrap-vcpkg.bat
~~~~~~~~~~~~~~~~~~~~~

- Add system environment variable for VCPKG_ROOT and VCPKG_DEFAULT_TRIPLET

~~~~~~~~~~~~~~~~~~~~~
    VCPKG_DEFAULT_TRIPLET=x64-windows-hypevr

    VCPKG_ROOT=XXX\vcpkg
~~~~~~~~~~~~~~~~~~~~~

- Creating a file named x64-windows-hypevr.cmake in XXX\vcpkg\triplets\

~~~~~~~~~~~~~~~~~~~~~
    set(VCPKG_TARGET_ARCHITECTURE x64)
    set(VCPKG_CRT_LINKAGE dynamic)

    if(PORT STREQUAL "glew")
      set(VCPKG_LIBRARY_LINKAGE static)
    else()
      set(VCPKG_LIBRARY_LINKAGE dynamic)
    endif()
~~~~~~~~~~~~~~~~~~~~~

- After installing external libraries, check if everything is correctly
  installed

~~~~~~~~~~~~~~~~~~~~~
    vcpkg list
~~~~~~~~~~~~~~~~~~~~~

- Vcpkg help

~~~~~~~~~~~~~~~~~~~~~
    vcpkg help
~~~~~~~~~~~~~~~~~~~~~

<br />

Setup coding style
==============

**For Linux: Using Sublime 3 as main IDE**

- Install prerequisites

~~~~~~~~~~~~~~~~~~~~~
    sudo apt-get install python
~~~~~~~~~~~~~~~~~~~~~

- Open Sublime

- Ctrl + `

- copy command from https://packagecontrol.io/installation

- press enter and restart Sublime

~~~~~~~~~~~~~~~~~~~~~
    sudo apt-get install python-pip

    sudo pip install cpplint
    sudo pip install cmakelint
~~~~~~~~~~~~~~~~~~~~~

- Open Sublime

- ctrl + shift + p

- select "package control: install package"

- search and install SublimeLinter

- search and install SublimeLinter-cpplint

- build the latest llvm/clang by following
http://clang.llvm.org/get_started.html (obtain latest release)

- ctrl + shift + p

- select "package control: install package"

- search and install clang format

- "Preferences"->"Package settings"->"Clang Format"->"Settings-User"

- add in (To find "clang-format", use "updatedb -> locate clang-format")

~~~~~~~~~~~~~~~~~~~~~
{
    "binary": "/path/to/clang/bin/clang-format",

    "style": "Custom",

    "format_on_save": true
}
~~~~~~~~~~~~~~~~~~~~~

- add following style to "Preferences"->"Package settings"->"Clang Format"->
"Custom Style-User"

~~~~~~~~~~~~~~~~~~~~~
{
    "BasedOnStyle": "Google",
    "AlignConsecutiveAssignments": true,
    "AlignTrailingComments": true,
    "AllowAllParametersOfDeclarationOnNextLine": true,
    "AllowShortCaseLabelsOnASingleLine": true,
    "AllowShortIfStatementsOnASingleLine": true,
    "AllowShortFunctionsOnASingleLine": "None",
    "AllowShortLoopsOnASingleLine": true,
    "BinPackArguments": false,
    "BinPackParameters": false,
    "BreakBeforeBraces": "Allman",
    "BreakConstructorInitializersBeforeComma": true,
    "ColumnLimit": 80
}
~~~~~~~~~~~~~~~~~~~~~

- ctrl + shift + p

- select "package control: install package"

- search and install SublimeLinter-contrib-cmakelint

- search and install EditorConfig, CMake, DoxyDoxygen (Evolution)

**For Windows: Using Visual Studio 2017 as main IDE**

- Download and Install the latest python, add path for all users

~~~~~~~~~~~~~~~~~~~~~
    easy_install cpplint
    easy_install cmakelint
~~~~~~~~~~~~~~~~~~~~~

- in Visual Studio, Tools -> External Tools -> Add. Name it cpplint

~~~~~~~~~~~~~~~~~~~~~
    Title             : cpplint
    Command           : cpplint.exe
    Arguments         : $(ItemPath)
    Initial Directory :
~~~~~~~~~~~~~~~~~~~~~

- Move the newly created cpplint to the very top

- in Visual Studio, Tools -> External Tools -> Add. Name it cmakelint

~~~~~~~~~~~~~~~~~~~~~
    Title             : cmakelint
    Command           : cmakelint.exe
    Arguments         : $(ItemPath)
    Initial Directory :
~~~~~~~~~~~~~~~~~~~~~

- Move the newly created cmakelint right beneath cpplint

- Download and install the latest clang-format plugin from
http://llvm.org/builds/

- Go to Tools->Options->LLVM/Clang->ClangFormat and put down "File" in the
Style option field.

- This will search for a .clang-format file in where the source is it's parent
directory

- Set "Format On Save" option to true

- Add in ";.cu;.cuh" into "File extensions"

- Install Visual Commander in Tools -> Extensions and Updates

- VCMD -> Extensions -> Add. Name it cpplint. Choose C# as a
language. Copy/Paste the following code. Save, Compile, Install.

~~~~~~~~~~~~~~~~~~~~~
    using EnvDTE;
    using EnvDTE80;

    public class E : VisualCommanderExt.IExtension
    {
      public void SetSite(EnvDTE80.DTE2 DTE_, Microsoft.VisualStudio.Shell.Package package)
      {
        DTE = DTE_;
        events = DTE.Events;
        documentEvents = events.DocumentEvents;
        documentEvents.DocumentSaved += OnDocumentSaved;
      }

      public void Close()
      {
        documentEvents.DocumentSaved -= OnDocumentSaved;
      }

      private void OnDocumentSaved(EnvDTE.Document doc)
      {
        if(doc.Language == "C/C++")
        {
          if(doc.Saved)
          {
            DTE.ExecuteCommand("Tools.ExternalCommand1");

            doc.Activate();
          }
          else
          {
            DTE.ExecuteCommand("File.SaveAll");
          }
        }
      }

      private EnvDTE80.DTE2 DTE;
      private EnvDTE.Events events;
      private EnvDTE.DocumentEvents documentEvents;
    }
~~~~~~~~~~~~~~~~~~~~~

- This will invoke cpplint and clang format whenever a c++ related file is
saved

- VCMD -> Extensions -> Add. Name it cmakelint. Choose C# as a
language. Copy/Paste the following code. Save, Compile, Install.

~~~~~~~~~~~~~~~~~~~~~
    using EnvDTE;
    using EnvDTE80;

    public class E : VisualCommanderExt.IExtension
    {
      public void SetSite(EnvDTE80.DTE2 DTE_, Microsoft.VisualStudio.Shell.Package package)
      {
        DTE = DTE_;
        events = DTE.Events;
        documentEvents = events.DocumentEvents;
        documentEvents.DocumentSaved += OnDocumentSaved;
      }

      public void Close()
      {
        documentEvents.DocumentSaved -= OnDocumentSaved;
      }

      private void OnDocumentSaved(EnvDTE.Document doc)
      {
        if(doc.Language == "Plain Text")
        {
          if(doc.Saved)
          {
            DTE.ExecuteCommand("Tools.ExternalCommand2");

            doc.Activate();
          }
          else
          {
            DTE.ExecuteCommand("File.SaveAll");
          }
        }
      }

      private EnvDTE80.DTE2 DTE;
      private EnvDTE.Events events;
      private EnvDTE.DocumentEvents documentEvents;
    }
~~~~~~~~~~~~~~~~~~~~~

- This will invoke cmakelint whenever a cmake related file is saved

- in Visual Studio, Tools -> Extensions and Updates -> search for
editorconfig and install it

<br />

External Libraries
==============

Doxygen
--------------

- To build with doxygen

~~~~~~~~~~~~~~~~~~~~~
   cmake ..
   make doc
~~~~~~~~~~~~~~~~~~~~~

**For Linux**

~~~~~~~~~~~~~~~~~~~~~
    sudo apt-get install doxygen texlive-full graphviz
~~~~~~~~~~~~~~~~~~~~~

**For Windows**

- Download and install the latest doxygen

- Download and install the latest MikTeX

- Download and install the latest Graphviz

- Manually add Graphviz to system PATH

**For Mac/OSX**

- install brew from http://brew.sh

~~~~~~~~~~~~~~~~~~~~~
    brew install doxygen

    brew install texlive-full

    brew install graphviz
~~~~~~~~~~~~~~~~~~~~~

CUDA
--------------

**For Linux**

- Download the latest cuda run file on NVIDIA's website

~~~~~~~~~~~~~~~~~~~~~
    sudo apt-get remove --purge cuda*

    chmod a+x cuda_x.x.xx_linux.run

    sudo ./cuda_x.x.xx_linux.run
~~~~~~~~~~~~~~~~~~~~~

- Just install cuda and the samples, don't install the driver

- Add system path

~~~~~~~~~~~~~~~~~~~~~
    vim ~/.bashrc
~~~~~~~~~~~~~~~~~~~~~

- export PATH=/usr/local/cuda/bin:$PATH

- export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

- Test samples

~~~~~~~~~~~~~~~~~~~~~
    cd NVIDIA_CUDA-x.x_Samples/bin/x86_64/linux/release

    ./deviceQuery

    ./bandwidthTest

    ./smokeParticles
~~~~~~~~~~~~~~~~~~~~~

**For Windows**

- Download the latest cuda

- Follow instructions on express installation

- Tell vcpkg cuda is installed

~~~~~~~~~~~~~~~~~~~~~
    vcpkg install cuda
~~~~~~~~~~~~~~~~~~~~~

**For Mac/OSX**

- Download the latest cuda

- Follow instructions (similar to linux install)

OpenCV
--------------

**For Linux**

~~~~~~~~~~~~~~~~~~~~~
    sudo apt-get install opencv-main opencv-dev opencv-libs
~~~~~~~~~~~~~~~~~~~~~

**For Windows**

~~~~~~~~~~~~~~~~~~~~~
    vcpkg install opencv
~~~~~~~~~~~~~~~~~~~~~

**For Mac/OSX**

- install brew from http://brew.sh

~~~~~~~~~~~~~~~~~~~~~
    brew install opencv3.2
~~~~~~~~~~~~~~~~~~~~~

- Follow brew after-installation instruction

- link opencv include, library, share directory from /usr/local/opt/opencv3 to
  /usr/local/include, /usr/local/lib, and /usr/local/share

Debugging
==============

GDB - For Linux
--------------

**Compile with GDB**

- Before using GDB, any source file needs to be
debugged must be compiled with -g flag

~~~~~~~~~~~~~~~~~~~~~
    g++ -g ...
~~~~~~~~~~~~~~~~~~~~~

**Basic operations**

 - Start gdb

~~~~~~~~~~~~~~~~~~~~~
    gdb executable_name
~~~~~~~~~~~~~~~~~~~~~

- Set break point

~~~~~~~~~~~~~~~~~~~~~
    b simple.cpp:147 (break on a line)
    b main (break upon entering a function)
    b simple.cpp:147 if x==0 (break on condition)
~~~~~~~~~~~~~~~~~~~~~

- get all break points information

~~~~~~~~~~~~~~~~~~~~~
    b info
~~~~~~~~~~~~~~~~~~~~~

- Disable/enable a break point. (break point ID can be found with "b info")

~~~~~~~~~~~~~~~~~~~~~
    disable 1
    enable 1
~~~~~~~~~~~~~~~~~~~~~

- Run the program

~~~~~~~~~~~~~~~~~~~~~
    r (run)
~~~~~~~~~~~~~~~~~~~~~

- Continue running after stop

~~~~~~~~~~~~~~~~~~~~~
    c (continue)
~~~~~~~~~~~~~~~~~~~~~

- step. Execute next line.

~~~~~~~~~~~~~~~~~~~~~
    n (next)
~~~~~~~~~~~~~~~~~~~~~

- Step in. Go inside a function call

~~~~~~~~~~~~~~~~~~~~~
    s (step)
~~~~~~~~~~~~~~~~~~~~~

- Step out of a function. Jump out of current function.

~~~~~~~~~~~~~~~~~~~~~
    finish
~~~~~~~~~~~~~~~~~~~~~

- Show variable value (temporary)

~~~~~~~~~~~~~~~~~~~~~
    print x
    print x.count
~~~~~~~~~~~~~~~~~~~~~

- Call a funciton

~~~~~~~~~~~~~~~~~~~~~
    print x.count()
~~~~~~~~~~~~~~~~~~~~~

- Display/undisplay variable (permanent)

~~~~~~~~~~~~~~~~~~~~~
    display x
    undisplay x
~~~~~~~~~~~~~~~~~~~~~

- Get the type of the variable

~~~~~~~~~~~~~~~~~~~~~
    ptype x
~~~~~~~~~~~~~~~~~~~~~

- Check memory.

~~~~~~~~~~~~~~~~~~~~~
    x/256x ptr
~~~~~~~~~~~~~~~~~~~~~

This command shows 256 bytes presented as hexidecimal
starting at address of ptr. Format can be changed to
(d)ecimal, (u)nsigned, etc

- Quit gdb

~~~~~~~~~~~~~~~~~~~~~
    quit
~~~~~~~~~~~~~~~~~~~~~

**Other**

- Find the source of "seg fault". At the most of the time,
GDB will automatically stop at the place where the
seg fault happened. This is a quick way to locate a seg fault.

~~~~~~~~~~~~~~~~~~~~~
    gdb a.out
    run
~~~~~~~~~~~~~~~~~~~~~

- cuda-gdb is capable of debugging cuda program running on
the gpu and its usage is exactly the same as normal gdb.

~~~~~~~~~~~~~~~~~~~~~
    cuda-gdb a.out
~~~~~~~~~~~~~~~~~~~~~

Visual Studio - For Windows
---------------------

Build project under debug configuration.

Set breakpoint by left-click column or press F9.

Press F5 to run and debug program.

During debugging.

- Step Into (F11).
  If the line contains a function call, Step Into executes only the call itself,
  then halts at the first line of code inside the function. Otherwise, Step Into
  executes the next statement.

- Step Over (F10).
  If the line contains a function call, Step Over executes the called function,
  then halts at the first line of code inside the calling function. Otherwise,
  Step Into executes the next statement.

- Step Out (Shift+F11).
  Step Out resumes execution of your code until the function returns, then
  breaks at the return point in the calling function.
