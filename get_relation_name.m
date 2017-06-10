function [relation_name] = get_relation_name(featnum, higher)

if featnum == 1
    if higher
        relation_name = 'larger';
    else
        relation_name = 'smaller';
    end
elseif featnum == 2
    if higher
        relation_name = 'fiercer';
    else
        relation_name = 'meeker';
    end
elseif featnum == 3
    if higher
        relation_name = 'smarter';
    else
        relation_name = 'stupider';
    end
elseif featnum == 4
    if higher
        relation_name = 'faster';
    else
        relation_name = 'slower';
    end
end