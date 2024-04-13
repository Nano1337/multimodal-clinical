
def get_model(args): 
    if args.model_type == "jlogits": 
        from mimic.joint_model import MultimodalMimicModel
    elif args.model_type == "ensemble":
        from mimic.ensemble_model import MultimodalMimicModel
    elif args.model_type == "jprobas":
        from mimic.joint_model_proba import MultimodalMimicModel
    else: 
        raise NotImplementedError("Model type not implemented")

    return MultimodalMimicModel(args)