import sys
import medicalgan.releases.mvp.mvp_gan as mvp_gan

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
        if release == "mvp":
            if task == "train":
                mvp_gan.train()
        #     if task == "generate":
        #         mvp.generate()
        # if release == "alpha":
        #     if task == "train":
        #         alpha.train()
        #     if task == "generate":
        #         alpha.generate()
