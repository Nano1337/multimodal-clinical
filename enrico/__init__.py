def get_model(args): 
    if args.model_type == "jlogits": 
        from enrico.joint_model import MultimodalEnricoModel
    elif args.model_type == "ensemble": 
        from enrico.ensemble_model import MultimodalEnricoModel
    elif args.model_type == "ensemble_counts": 
        from enrico.ensemble_model_counts import MultimodalEnricoModel
    elif args.model_type == "jlogits_counts": 
        from enrico.joint_model_counts import MultimodalEnricoModel
    elif args.model_type == "ensemble_vicreg": 
        from enrico.ensemble_model_vicreg import MultimodalEnricoModel
    else: 
        raise NotImplementedError("Model type not implemented")
    
    return MultimodalEnricoModel(args)
