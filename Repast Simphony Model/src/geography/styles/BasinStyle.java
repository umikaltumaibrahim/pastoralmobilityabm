package geography.styles;

import java.awt.Color;

import geography.agents.Basin;
import gov.nasa.worldwind.render.Material;
import repast.simphony.visualization.gis3D.style.DefaultSurfaceShapeStyle;
import repast.simphony.visualization.gis3D.style.SurfaceShapeStyle;

public class BasinStyle extends DefaultSurfaceShapeStyle<Basin> implements SurfaceShapeStyle<Basin> {
    
    public BasinStyle() {
        // Default constructor - can be empty or set global properties
    }
    
    public Material getFillMaterial(Basin basin) {
        // Semi-transparent blue fill
        return new Material(new Color(0, 0, 255, 128)); // RGBA: Blue with 50% opacity
    }
    
    public Material getLineMaterial(Basin basin) {
        // Black outline
        return Material.BLACK;
    }
    
    @Override
    public double getLineWidth(Basin basin) {
        // Outline thickness
        return 2.0;
    }

    public boolean isFilled(Basin basin) {
        return true; // Enable fill
    }
    
    public boolean isLineEnabled(Basin basin) {
        return true; // Enable outline
    }
}