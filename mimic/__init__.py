
def get_model(args): 
    if args.model_type == "jlogits": 
        from mimic.joint_model import MultimodalMimicModel
    elif args.model_type == "ensemble":
        from mimic.ensemble_model import MultimodalMimicModel
    elif args.model_type == "jprobas":
        from mimic.joint_model_proba import MultimodalMimicModel
    elif args.model_type == "ogm_ge": 
        from mimic.ogm_ge_model import MultimodalMimicModel
    elif args.model_type == "qmf": 
        from mimic.qmf_model import MultimodalMimicModel
    else: 
        raise NotImplementedError("Model type not implemented")

    return MultimodalMimicModel(args)