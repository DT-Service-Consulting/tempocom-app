def labs_card(lab:dict):
     
    internal_style={
       True:{
          'color':'#60B2E0',
          'text':'PRIVATE'
       },
       False:{
          'color':'grey',
          'text':'PUBLIC'
       }
    }

    return f"""
        <div style='border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); text-align: center; margin: 10px 0;
        height: 200px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <h3>{lab.get('title','Title')}</h3>
                <p>{lab.get('description','Description')}</p>
                <div style='color: {internal_style[lab['private']]['color']}; font-weight: bold;'>{internal_style[lab['private']]['text']}</div>
            </div>
        </div>
        """

def digital_twin_card():
    return f"""
    <div style='background-color: #f0e6ff; border-radius: 10px; padding: 20px; border-left: 5px solid #6a0dad; margin: 20px 0;'>
        <h3 style='color: #6a0dad; text-align: center;'>The Big Digital Twin</h3>
        <p style='text-align: center; font-size: 16px;'>
            Access to advanced and exclusive features to optimize your railway system.
        </p>
        <div style='display: flex; justify-content: center; margin-top: 20px;'>
            <ul style='list-style-type: none; padding-left: 0;'>
                <li style='margin-bottom: 10px;'>✓ Advanced predictive maintenance</li>
                <li style='margin-bottom: 10px;'>✓ Causality-based Decision Aids</li>
                <li style='margin-bottom: 10px;'>✓ Scenarios simulation</li>
            </ul>
        </div>
        <div style='display: flex; justify-content: center; margin-top: 20px;'>
            <button disabled style='background-color: #6a0dad; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: not-allowed; opacity: 0.8; background-image: linear-gradient(45deg, #6a0dad 25%, #8a2be2 25%, #8a2be2 50%, #6a0dad 50%, #6a0dad 75%, #8a2be2 75%, #8a2be2 100%); background-size: 20px 20px;'>
                2027
            </button>
        </div>
    </div>"""