# Installation


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

This new version of NA³ is currently available via the [Python Package
Index](https://pypi.org/project/neuralactivitycubic/) or via
[GitHub](https://github.com/Indoc-Research/NeuralActivityCubic). To be
able to install and run NA³ on your system you currently need to run a
few commands in the terminal, as the fully GUI-based installation will
be available only soon. But don´t worry, we´ll take you through the
process step by step!

<div>

> **Note**
>
> We´re still putting a few finishing touches onto the new
> implementation of NA³, so please be aware that this version remains
> under active development and should not yet be considered as a stable
> release.

</div>

<div>

> **Tip**
>
> We´re currently working on a paper describing our work in more
> details - make sure you stay tuned and regularly check these docs for
> updates!

</div>

## 1. Install Anaconda

Though not strictly required, we highly recommend installing NA³ in a
virtual Python environment that you setup via Anaconda - especially for
users with limited (or even no) programming expertise. Please download
the corresponding distribution for your operating system
[here](https://www.anaconda.com/download) and follow the respective
installation wizard to install Anaconda on your system.

## 2. Create and activate a new virtual environment

Next, we´ll need a terminal to install NA³ with just a few commands. For
this, we suggest using the tool Anaconda Prompt that came with the
installation of Anaconda (but if you prefer to use a different terminal,
please feel free to use it - the only requirement is that `conda` is
accessible, which comes per default in Anaconda Prompt). In the terminal
(e.g. in Anaconda Prompt), please type the following command and execute
it by hitting the Return key on your keyboard. It will create a new
virtual environment that we can use to install NA³, using `conda`:

``` python
conda create --name na3 -y python=3.11
```

Once the setup of the environment has concluded, you should see a
message like the following, informing you how to activate the
environment:

[./media/create_conda_env.mp4](./media/create_conda_env.mp4)

We´ll do exactly as suggested to activate the newly created virtual
environment to continue the installation of NA³. Thus, please run the
following command in the Anaconda Prompt terminal:

``` python
conda activate na3
```

You should now see that the input line in the Anaconda Prompt terminal
starts with `(na3)` - indicating that you successfully switched to this
virtual environment.

## 3. Install the `neuralactivitycubic` Python package:

With the activated na3 environment, please complete the installation of
NA³ by running on final command in the Anaconda Prompt terminal:

``` python
pip install neuralactivitycubic
```

Upon executing this command, you will see a lot of output and activity
in the terminal while it downloads the source code of NA³ and it´s
dependencies, and installs them all in the virtual environment on your
system. You know that the installation has finished, when the input line
starting with `(na3)` appears again. The final few messages that are
shown in Anaconda Prompt will then likely look similar to this:

<figure>
<img src="./media/successfully_installed_na3.png"
alt="Output after successful installation of neuralactivitycubic." />
<figcaption aria-hidden="true">Output after successful installation of
neuralactivitycubic.</figcaption>
</figure>

## 4. Done!

Congratulations! That´s already everything you need to do to install NA³
on your system. You can now use NA³ to analyze your data and hopefully
start a journey to the next scientific breakthrough!

If you´d like to get some more information on how to use NA³, please
check out the following chapters in this documentation:

- [Using NA³ via the
  GUI](https://indoc-research.github.io/NeuralActivityCubic/using_the_gui.html)
- \[Using NA³ via the API\]

## We´d love to hear your Feedback:

You are using NA³ for your research - or would like to do so, but
there´s that one cool feature missing for you? That´s great! We´d love
to hear your feedback, feature requests, or bug reports to keep
improving NA³ - please use [this form on
GitHub](https://github.com/Indoc-Research/NeuralActivityCubic/issues/new)
to submit it, we appreciate it a lot!

Please feel free to also explore all other [Research
Software](https://github.com/Indoc-Research/) that our not-for-profit
organization has developed, or learn more about us and what we do on our
[website](https://www.indocresearch.eu/). If you are interested to
partner with us for a similar collaboration on your Research Software,
or to develop it with or for you from scratch, don´t hesitate and drop
us an [email](mailto:ddoll@indocresearch.org) - we´re always looking for
new opportunities to catalyze science be delivering cutting-edge
technology tailored to the specific research challenges of our
collaborators!

<figure>
<img
src="data:image/png;base64,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"
alt="Indoc Research Europe gGmbH" />
<figcaption aria-hidden="true">Indoc Research Europe gGmbH</figcaption>
</figure>