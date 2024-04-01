
def get_model(args):
    # model training type
    if args.model_type == "jlogits":
        from cremad.joint_model import MultimodalCremadModel
    elif args.model_type == "ensemble":
        from cremad.ensemble_model import MultimodalCremadModel
    elif args.model_type == "jprobas":
        from cremad.joint_model_proba import MultimodalCremadModel
    elif args.model_type == "ogm_ge": 
        from cremad.joint_model_ogm_ge import MultimodalCremadModel
    elif args.model_type == "ensemble_ogm_ge": 
        from cremad.ensemble_model_noised import MultimodalCremadModel
    elif args.model_type == "qmf": 
        from cremad.joint_model_qmf import MultimodalCremadModel
    elif args.model_type == "qmf_ablate": 
        from cremad.joint_model_qmf_ablate import MultimodalCremadModel
    elif args.model_type == "qmf_ablate_Ljoint": 
        from cremad.joint_model_qmf_ablate_Ljoint import MultimodalCremadModel
    elif args.model_type == "qmf_ablate_Lunimodal": 
        from cremad.joint_model_qmf_ablate_Lunimodal import MultimodalCremadModel
    elif args.model_type == "ogm_ge_lreg": 
        from cremad.joint_model_ogm_ge_lreg import MultimodalCremadModel
    elif args.model_type == "biasampler": 
        from cremad.ensemble_model_biasampler import MultimodalCremadModel
    else:   
        raise NotImplementedError("Model type not implemented")

    # get model
    return MultimodalCremadModel(args)

