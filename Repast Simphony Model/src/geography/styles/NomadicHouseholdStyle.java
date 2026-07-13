package geography.styles;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

import geography.agents.NomadicHousehold;
import gov.nasa.worldwind.WorldWind;
import gov.nasa.worldwind.avlist.AVKey;
import gov.nasa.worldwind.render.BasicWWTexture;
import gov.nasa.worldwind.render.Material;
import gov.nasa.worldwind.render.Offset;
import gov.nasa.worldwind.render.PatternFactory;
import gov.nasa.worldwind.render.WWTexture;
import repast.simphony.visualization.gis3D.PlaceMark;
import repast.simphony.visualization.gis3D.style.MarkStyle;

/**
 * Updated Style for Version 6.
 * - Optimized for dynamic agent populations (8000+).
 * - Removed dependencies on 'isMoving' and 'ClanId'.
 * - Visualizes Agents as Blue Circles.
 */
public class NomadicHouseholdStyle implements MarkStyle<NomadicHousehold> {
    
    private Offset labelOffset;
    private Map<String, WWTexture> textureMap;
    
    public NomadicHouseholdStyle() {
        // Offset label slightly above and to the right of the mark
        labelOffset = new Offset(1.2d, 0.6d, AVKey.FRACTION, AVKey.FRACTION);
        textureMap = new HashMap<>();
        
        createTextures();
    }
    
    private void createTextures() {
        // Standard State - Blue Circle
        // Kept small (15x15) for high density population
        BufferedImage circleImage = PatternFactory.createPattern(
            PatternFactory.PATTERN_CIRCLE, 
            new Dimension(15, 15), 
            0.8f, 
            Color.BLUE
        );
        textureMap.put("default", new BasicWWTexture(circleImage));
    }
    
    @Override
    public PlaceMark getPlaceMark(NomadicHousehold agent, PlaceMark mark) {
        if (mark == null) {
            mark = new PlaceMark();
        }
        // Use relative to ground to ensure they sit on the terrain/raster
        mark.setAltitudeMode(WorldWind.RELATIVE_TO_GROUND);
        mark.setLineEnabled(false); // Disable leader lines for performance
        return mark;
    }
    
    @Override
    public double getElevation(NomadicHousehold agent) {
        // Flat elevation
        return 0;
    }
    
    @Override
    public WWTexture getTexture(NomadicHousehold agent, WWTexture texture) {
        // Return the single default texture for all agents in V6
        return textureMap.get("default");
    }
    
    @Override
    public double getScale(NomadicHousehold agent) {
        // Fixed scale 
        return 0.5; 
    }

    @Override
    public String getLabel(NomadicHousehold agent) {
        // Return empty string for performance with 8000+ agents.
        return ""; 
    }

    @Override
    public Color getLabelColor(NomadicHousehold agent) {
        return Color.WHITE; // Default color since Clan logic is removed
    }
    
    @Override
    public Offset getLabelOffset(NomadicHousehold agent) {
        return labelOffset;
    }

    @Override
    public Font getLabelFont(NomadicHousehold agent) {
        return new Font("Arial", Font.PLAIN, 10);
    }

    @Override
    public double getLineWidth(NomadicHousehold agent) {
        return 0;
    }

    @Override
    public Material getLineMaterial(NomadicHousehold agent, Material lineMaterial) {
        return new Material(Color.GRAY); // Default material
    }

    @Override
    public Offset getIconOffset(NomadicHousehold agent) {
        return Offset.CENTER;
    }

    @Override
    public double getHeading(NomadicHousehold agent) {
        return 0;
    }
}