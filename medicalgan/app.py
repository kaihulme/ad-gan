import sys
import medicalgan.releases.mnist.mnist_gan as mnist_gan
import medicalgan.releases.fmri_tumour.fmri_tumour_cnn as fmri_tumour_cnn
import medicalgan.releases.adni_alzheimers.adni_alzheimers_cnn as adni_alzheimers_cnn


def run():
    """
    Generation of medical images using Generative Adversarial Networks.
    """
    args = sys.argv
    if not len(args) == 3:
        print("\nPlease specify release and task:\
               \n>  medicalgan [release] [task]\n")
    else:
        release, task = args[1], args[2]
        if release == "mnist":
            if task == "train":
                mnist_gan.train()
            # if task == "generate":
                # mvp.generate()
        if release == "fmri_tumour":
            # if task == "train":
                # fmri_tumour_gan.train()
            # if task == "generate":
                # fmri_tumour_gan.generate()
            if task == "detect":
                fmri_tumour_cnn.train()
        # if release == "nih_chest_xray":
            # if task == "train":
                # nih_chest_xray_gan.train()
            # if task == "generate":
                # nih_chest_xray_gan.generate()
            # if task == "detect":
            #     nih_chest_xray_cnn.train()
        if release == "adni_alzheimers":
            # if task == "train":
                # adni_alzheimers_gan.train()
            # if task == "generate":
                # adni_alzheimers_gan.generate()
            if task == "detect":
                adni_alzheimers_cnn.train()
