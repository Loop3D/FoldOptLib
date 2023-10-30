import ipywidgets as widgets
from IPython.display import display, clear_output


def create_mu_widgets():
    mu_widgets = [
        widgets.FloatText(value=0, description=f'mu[{i}]') for i in range(3)
    ]
    return widgets.HBox(mu_widgets)


def create_value_widgets(constraint_dropdown, sub_constraint_dropdown):
    selected_constraint = constraint_dropdown.value
    selected_sub_constraint = sub_constraint_dropdown.value
    if selected_constraint == 'fold_axial_surface' and selected_sub_constraint == 'axial_surface':
        mu_widget = create_mu_widgets()
        return {
            'mu': mu_widget,
            'kappa': widgets.FloatText(value=5, description='kappa'),
            'w': widgets.FloatText(value=1, description='w')
        }
    else:
        mu_widget = widgets.FloatText(value=0, description='mu')
    return {
        'mu': mu_widget,
        'sigma': widgets.FloatText(value=0, description='sigma'),
        'w': widgets.FloatText(value=1, description='w')
    }


def on_add_button_click(constraint_dropdown, sub_constraint_dropdown):
    selected_constraint = constraint_dropdown.value
    selected_sub_constraint = sub_constraint_dropdown.value
    values = {}
    for k, v in value_widgets.items():
        if isinstance(v, widgets.HBox):
            # Assuming mu is the only HBox and is always composed of three FloatText widgets
            values[k] = [w.value for w in v.children]
        else:
            values[k] = v.value
    dict_structure[selected_constraint][selected_sub_constraint] = values
    with output:
        clear_output()
        print(dict_structure)


def on_constraint_change(change, sub_constraint_dropdown):
    new_value = change.get('new', None)
    sub_constraint_dropdown.options = sub_constraints.get(new_value, [])
    # Only call on_sub_constraint_change if form is defined
    if 'form' in globals():
        on_sub_constraint_change({'new': sub_constraint_dropdown.value})


def on_sub_constraint_change(constraint_dropdown, sub_constraint_dropdown):
    global value_widgets
    value_widgets = create_value_widgets()
    form.children = [constraint_dropdown, sub_constraint_dropdown] + list(value_widgets.values()) + [add_button, output]


def display_dict_selection(sub_constraints):
    # Dropdown for constraints
    constraint_dropdown = widgets.Dropdown(options=list(sub_constraints.keys()), description='Major Constraint:')
    constraint_dropdown.observe(on_constraint_change, names='value')

    # Dropdown for sub-constraints
    sub_constraint_dropdown = widgets.Dropdown(description='Sub-Constraint:')
    sub_constraint_dropdown.observe(on_sub_constraint_change, names='value')

    # Button to add the details
    add_button = widgets.Button(description="Add Details")
    add_button.on_click(on_add_button_click)

    # Output widget to display the generated dictionary
    output = widgets.Output()

    # Initial value widgets
    value_widgets = create_value_widgets(constraint_dropdown, sub_constraint_dropdown)

    # Form to hold all the widgets
    form = widgets.VBox(
        [constraint_dropdown, sub_constraint_dropdown] + list(value_widgets.values()) + [add_button, output])

    # Initial setup
    on_constraint_change({'new': constraint_dropdown.value})

    display(form)
