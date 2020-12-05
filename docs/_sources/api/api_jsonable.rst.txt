Jsonable API
============


.. automodule:: pheres._jsonable
    :members:
    :show-inheritance:
    :exclude-members: jsonable

    .. autoclass:: jsonable(cls:type=None, /, after:Union[str, Iterable[str]]=(), internal:str=None, only_marked:bool=False)
        :members:

        .. attribute:: Value[T, ...]

            Shortcut for ``jsonable[Union[T, ...]]``
        
        .. attribute:: Array[T, ...]
            
            Shortcut for ``jsonable[Tuple[T, ...]]``
        
        .. attribute:: Dict[T, ...]
            
            Shortcut for ``jsonable[Dict[str, Union[T, ...]]]``